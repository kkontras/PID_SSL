"""
Contrastive pretraining only (no classification head / label usage).
Implements Rung 7A of the experimental plan.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset_v3 import V3Config, V3DatasetGenerator, ALL_CONFIGS
from models.model_v3 import ModelV3Config, MultimodalContrastiveModel
from losses.combined import combined_loss
from losses.masked_pred import random_feature_mask
from utils.metrics import representation_diagnostics


def _save_history_artifacts(history: List[Dict], save_dir: str) -> None:
    csv_path = os.path.join(save_dir, "history.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(7, 4.5))
    plt.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train_loss")
    if "val_loss" in history[0]:
        plt.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Pretraining loss curves")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=140)
    plt.close()


def _make_noisy_views(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    noise_std: float = 0.1,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return (
        (x1 + noise_std * torch.randn_like(x1), x2 + noise_std * torch.randn_like(x2), x3 + noise_std * torch.randn_like(x3)),
        (x1 + noise_std * torch.randn_like(x1), x2 + noise_std * torch.randn_like(x2), x3 + noise_std * torch.randn_like(x3)),
    )


def _build_ssl_state(
    model: MultimodalContrastiveModel,
    method: str,
    x1_b: torch.Tensor,
    x2_b: torch.Tensor,
    x3_b: torch.Tensor,
    mask_ratio: float,
    n_mask_samples: int,
    view_noise_std: float,
    ema_model: Optional[MultimodalContrastiveModel] = None,
) -> Dict[str, object]:
    state: Dict[str, object] = {
        "base_out": None,
        "masked_out": None,
        "mask1": None,
        "mask2": None,
        "mask3": None,
        "z1_a": None,
        "z2_a": None,
        "z3_a": None,
        "z1_b": None,
        "z2_b": None,
        "z3_b": None,
        "zf_a": None,
        "zf_b": None,
        "zf_a_masked_list": None,
        "zf_b_masked_list": None,
    }

    if method in ("masked_raw", "masked_emb"):
        x1_m, mask1 = random_feature_mask(x1_b, mask_ratio)
        x2_m, mask2 = random_feature_mask(x2_b, mask_ratio)
        x3_m, mask3 = random_feature_mask(x3_b, mask_ratio)
        out = model.forward(x1_m, x2_m, x3_m, x1_full=x1_b, x2_full=x2_b, x3_full=x3_b)
        if method == "masked_emb":
            assert ema_model is not None
            teacher_out = ema_model.forward(x1_b, x2_b, x3_b)
            out["h1_teacher"] = teacher_out["h1"].detach()
            out["h2_teacher"] = teacher_out["h2"].detach()
            out["h3_teacher"] = teacher_out["h3"].detach()

        # Cross-modal TF: substitute per-modality reps with cross-attended versions
        if "cm_h1_masked" in out:
            if method == "masked_emb":
                # Student: masked inputs processed by cross-modal TF
                out["h1_masked"] = out["cm_h1_masked"]
                out["h2_masked"] = out["cm_h2_masked"]
                out["h3_masked"] = out["cm_h3_masked"]
                # Teacher: full inputs processed by cross-modal TF (stop-grad)
                out["h1_teacher"] = out["cm_h1_full"].detach()
                out["h2_teacher"] = out["cm_h2_full"].detach()
                out["h3_teacher"] = out["cm_h3_full"].detach()
            elif method == "masked_raw":
                # Project cross-modal representations for the raw decoder
                out["z1_masked"] = model.proj_heads[0](out["cm_h1_masked"])
                out["z2_masked"] = model.proj_heads[1](out["cm_h2_masked"])
                out["z3_masked"] = model.proj_heads[2](out["cm_h3_masked"])

        state["base_out"] = out
        state["masked_out"] = out
        state["mask1"] = mask1
        state["mask2"] = mask2
        state["mask3"] = mask3
        return state

    if method in ("simclr", "comm", "infmask"):
        (x1_a, x2_a, x3_a), (x1_b_aug, x2_b_aug, x3_b_aug) = _make_noisy_views(
            x1_b, x2_b, x3_b, noise_std=view_noise_std
        )
        out_a = model.forward(x1_a, x2_a, x3_a)
        out_b = model.forward(x1_b_aug, x2_b_aug, x3_b_aug)
        state["base_out"] = out_a
        state["z1_a"] = out_a["z1"]
        state["z2_a"] = out_a["z2"]
        state["z3_a"] = out_a["z3"]
        state["z1_b"] = out_b["z1"]
        state["z2_b"] = out_b["z2"]
        state["z3_b"] = out_b["z3"]

        if method in ("comm", "infmask"):
            state["zf_a"] = model.fuse_all(out_a["z1"], out_a["z2"], out_a["z3"])
            state["zf_b"] = model.fuse_all(out_b["z1"], out_b["z2"], out_b["z3"])

        if method == "infmask":
            masked_a = []
            masked_b = []
            for _ in range(n_mask_samples):
                x1_a_m, _ = random_feature_mask(x1_a, mask_ratio)
                x2_a_m, _ = random_feature_mask(x2_a, mask_ratio)
                x3_a_m, _ = random_feature_mask(x3_a, mask_ratio)
                out_a_m = model.forward(x1_a_m, x2_a_m, x3_a_m)
                masked_a.append(model.fuse_all(out_a_m["z1"], out_a_m["z2"], out_a_m["z3"]))

                x1_b_m, _ = random_feature_mask(x1_b_aug, mask_ratio)
                x2_b_m, _ = random_feature_mask(x2_b_aug, mask_ratio)
                x3_b_m, _ = random_feature_mask(x3_b_aug, mask_ratio)
                out_b_m = model.forward(x1_b_m, x2_b_m, x3_b_m)
                masked_b.append(model.fuse_all(out_b_m["z1"], out_b_m["z2"], out_b_m["z3"]))
            state["zf_a_masked_list"] = tuple(masked_a)
            state["zf_b_masked_list"] = tuple(masked_b)
        return state

    state["base_out"] = model.forward(x1_b, x2_b, x3_b)
    return state


def _compute_val_loss_batched(
    model: MultimodalContrastiveModel,
    ema_model: Optional[MultimodalContrastiveModel],
    method: str,
    x1_va: torch.Tensor,
    x2_va: torch.Tensor,
    x3_va: torch.Tensor,
    temperature: float,
    lambda_contr: float,
    device: str,
    eval_batch_size: int,
    mask_ratio: float = 0.5,
    masked_emb_var_weight: float = 1.0,
    lambda_mask: float = 1.0,
    n_mask_samples: int = 1,
    view_noise_std: float = 0.1,
    triangle_alpha: float = 0.0,
    confu_fuse_weight: float = 0.5,
) -> float:
    losses: List[float] = []
    for start in range(0, x1_va.shape[0], eval_batch_size):
        x1_b = x1_va[start:start + eval_batch_size].to(device)
        x2_b = x2_va[start:start + eval_batch_size].to(device)
        x3_b = x3_va[start:start + eval_batch_size].to(device)

        state = _build_ssl_state(
            model=model,
            method=method,
            x1_b=x1_b,
            x2_b=x2_b,
            x3_b=x3_b,
            mask_ratio=mask_ratio,
            n_mask_samples=n_mask_samples,
            view_noise_std=view_noise_std,
            ema_model=ema_model,
        )
        out_val = state["base_out"]
        z1_va_b, z2_va_b, z3_va_b = out_val["z1"], out_val["z2"], out_val["z3"]
        fused_val = model.fuse(z1_va_b, z2_va_b, z3_va_b) if method == "confu" else None

        val_loss, _ = combined_loss(
            method=method,
            z1=z1_va_b,
            z2=z2_va_b,
            z3=z3_va_b,
            fused=fused_val,
            z1_a=state["z1_a"],
            z1_b=state["z1_b"],
            z2_a=state["z2_a"],
            z2_b=state["z2_b"],
            z3_a=state["z3_a"],
            z3_b=state["z3_b"],
            zf_a=state["zf_a"],
            zf_b=state["zf_b"],
            zf_a_masked_list=state["zf_a_masked_list"],
            zf_b_masked_list=state["zf_b_masked_list"],
            temperature=temperature,
            lambda_contr=lambda_contr,
            lambda_mask=lambda_mask,
            triangle_alpha=triangle_alpha,
            confu_pair_weight=1.0 - confu_fuse_weight,
            confu_fuse_weight=confu_fuse_weight,
            masked_out=state["masked_out"],
            x1=x1_b if method == "masked_raw" else None,
            x2=x2_b if method == "masked_raw" else None,
            x3=x3_b if method == "masked_raw" else None,
            mask1=state["mask1"],
            mask2=state["mask2"],
            mask3=state["mask3"],
            masked_emb_var_weight=masked_emb_var_weight,
        )
        losses.append(float(val_loss.detach().cpu()))
    return float(np.mean(losses))


def _update_ema(teacher: MultimodalContrastiveModel, student: MultimodalContrastiveModel, momentum: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_buf, student_buf in zip(teacher.buffers(), student.buffers()):
            teacher_buf.copy_(student_buf)


def pretrain(
    method: str,
    config_name: str,
    Q: int = 7,
    D: int = 44,
    D_info: int = 4,
    n_train: int = 50000,
    n_val: int = 5000,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    d_z: int = 64,
    n_patches: int = 4,
    temperature: float = 0.07,
    lambda_contr: float = 1.0,
    batch_size: int = 512,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    save_dir: Optional[str] = None,
    log_every: int = 10,
    mask_ratio: float = 0.5,
    ema_momentum: float = 0.996,
    masked_emb_var_weight: float = 1.0,
    lambda_mask: float = 1.0,
    n_mask_samples: int = 1,
    view_noise_std: float = 0.1,
    triangle_alpha: float = 0.0,
    confu_fuse_weight: float = 0.5,
    use_cross_modal_tf: bool = False,
    cross_modal_n_layers: int = 2,
    cross_modal_n_heads: int = 4,
    cross_modal_d_ff: int = 128,
) -> List[Dict]:
    """Run contrastive pretraining and return training history."""
    print("=" * 72)
    print(f"Pretrain | method={method} | config={config_name} | device={device} | seed={seed}")
    print("=" * 72)
    torch.manual_seed(seed)
    np.random.seed(seed)

    atoms = ALL_CONFIGS[config_name]
    data_cfg = V3Config(Q=Q, D=D, D_info=D_info, active_atoms=atoms, n_samples=n_train + n_val, seed=seed)
    gen = V3DatasetGenerator(data_cfg)

    print(f"Generating dataset: train={n_train} val={n_val} Q={Q} D={D} D_info={D_info}")
    data = gen.generate(n_train + n_val)
    x1_all = torch.from_numpy(data["x1"]).float()
    x2_all = torch.from_numpy(data["x2"]).float()
    x3_all = torch.from_numpy(data["x3"]).float()
    x1_tr, x1_va = x1_all[:n_train], x1_all[n_train:]
    x2_tr, x2_va = x2_all[:n_train], x2_all[n_train:]
    x3_tr, x3_va = x3_all[:n_train], x3_all[n_train:]

    model_cfg = ModelV3Config(
        d_input=D, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_model * 2, n_patches=n_patches, d_z=d_z, proj_hidden=d_model * 2,
        fused_hidden=d_model * 2,
        n_classes=data_cfg.n_classes(),
        use_cross_modal_tf=use_cross_modal_tf,
        cross_modal_n_layers=cross_modal_n_layers,
        cross_modal_n_heads=cross_modal_n_heads,
        cross_modal_d_ff=cross_modal_d_ff,
    )
    model = MultimodalContrastiveModel(model_cfg).to(device)
    ema_model: Optional[MultimodalContrastiveModel] = None
    if method == "masked_emb":
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    n_steps = (n_train + batch_size - 1) // batch_size
    eval_batch_size = min(batch_size, 256)
    print(
        f"Model/training: batch_size={batch_size} epochs={epochs} "
        f"steps_per_epoch={n_steps} lr={lr} weight_decay={weight_decay} eval_batch_size={eval_batch_size}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        perm = torch.randperm(n_train)
        for step in range(n_steps):
            idx = perm[step * batch_size: (step + 1) * batch_size]
            x1_b = x1_tr[idx].to(device)
            x2_b = x2_tr[idx].to(device)
            x3_b = x3_tr[idx].to(device)

            state = _build_ssl_state(
                model=model,
                method=method,
                x1_b=x1_b,
                x2_b=x2_b,
                x3_b=x3_b,
                mask_ratio=mask_ratio,
                n_mask_samples=n_mask_samples,
                view_noise_std=view_noise_std,
                ema_model=ema_model,
            )
            out = state["base_out"]
            z1, z2, z3 = out["z1"], out["z2"], out["z3"]

            fused = None
            if method == "confu":
                fused = model.fuse(z1, z2, z3)

            loss, metrics = combined_loss(
                method=method, z1=z1, z2=z2, z3=z3,
                fused=fused,
                z1_a=state["z1_a"], z1_b=state["z1_b"],
                z2_a=state["z2_a"], z2_b=state["z2_b"],
                z3_a=state["z3_a"], z3_b=state["z3_b"],
                zf_a=state["zf_a"], zf_b=state["zf_b"],
                zf_a_masked_list=state["zf_a_masked_list"],
                zf_b_masked_list=state["zf_b_masked_list"],
                temperature=temperature,
                lambda_contr=lambda_contr,
                lambda_mask=lambda_mask,
                triangle_alpha=triangle_alpha,
                confu_pair_weight=1.0 - confu_fuse_weight,
                confu_fuse_weight=confu_fuse_weight,
                masked_out=state["masked_out"],
                x1=x1_b if method == "masked_raw" else None,
                x2=x2_b if method == "masked_raw" else None,
                x3=x3_b if method == "masked_raw" else None,
                mask1=state["mask1"],
                mask2=state["mask2"],
                mask3=state["mask3"],
                masked_emb_var_weight=masked_emb_var_weight,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            if method == "masked_emb":
                assert ema_model is not None
                _update_ema(ema_model, model, ema_momentum)
            epoch_losses.append(float(loss.detach().cpu()))

        scheduler.step()
        avg_loss = np.mean(epoch_losses)

        model.eval()
        with torch.no_grad():
            val_loss = _compute_val_loss_batched(
                model=model,
                ema_model=ema_model,
                method=method,
                x1_va=x1_va,
                x2_va=x2_va,
                x3_va=x3_va,
                temperature=temperature,
                lambda_contr=lambda_contr,
                device=device,
                eval_batch_size=eval_batch_size,
                mask_ratio=mask_ratio,
                masked_emb_var_weight=masked_emb_var_weight,
                lambda_mask=lambda_mask,
                n_mask_samples=n_mask_samples,
                view_noise_std=view_noise_std,
                triangle_alpha=triangle_alpha,
                confu_fuse_weight=confu_fuse_weight,
            )

        row = {"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss}
        if epoch % log_every == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | train_loss={avg_loss:.4f} val_loss={row['val_loss']:.4f}")
        history.append(row)

        # Save checkpoint
        if save_dir and epoch % 50 == 0:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "model_cfg": model_cfg,
                "method": method,
                "config_name": config_name,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "epoch": epochs,
            "model_state": model.state_dict(),
            "model_cfg": model_cfg,
            "method": method,
            "config_name": config_name,
        }, os.path.join(save_dir, "final.pt"))
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f)
        _save_history_artifacts(history, save_dir)
        print(f"Saved pretrain artifacts to: {save_dir}")

    best_val = min(row["val_loss"] for row in history)
    final_row = history[-1]
    print(
        f"Completed pretrain | final_train_loss={final_row['train_loss']:.4f} "
        f"| final_val_loss={final_row['val_loss']:.4f} | best_val_loss={best_val:.4f}"
    )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="pairwise_nce",
                        choices=["none", "simclr", "pairwise_nce", "triangle", "confu",
                                 "masked_raw", "masked_emb", "comm", "infmask"])
    parser.add_argument("--config", default="A4")
    parser.add_argument("--Q", type=int, default=7)
    parser.add_argument("--D", type=int, default=44)
    parser.add_argument("--D_info", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_z", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--lambda_contr", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--ema_momentum", type=float, default=0.996)
    parser.add_argument("--masked_emb_var_weight", type=float, default=1.0)
    parser.add_argument("--lambda_mask", type=float, default=1.0)
    parser.add_argument("--n_mask_samples", type=int, default=1)
    parser.add_argument("--view_noise_std", type=float, default=0.1)
    parser.add_argument("--triangle_alpha", type=float, default=0.0)
    parser.add_argument("--confu_fuse_weight", type=float, default=0.5)
    parser.add_argument("--use_cross_modal_tf", type=int, default=0)  # 0/1 for condor compat
    parser.add_argument("--cross_modal_n_layers", type=int, default=2)
    parser.add_argument("--cross_modal_n_heads", type=int, default=4)
    parser.add_argument("--cross_modal_d_ff", type=int, default=128)
    args = parser.parse_args()

    pretrain(
        method=args.method,
        config_name=args.config,
        Q=args.Q, D=args.D, D_info=args.D_info,
        n_train=args.n_train,
        d_model=args.d_model, d_z=args.d_z, n_layers=args.n_layers,
        temperature=args.tau, lambda_contr=args.lambda_contr,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed, device=args.device, save_dir=args.save_dir,
        mask_ratio=args.mask_ratio,
        ema_momentum=args.ema_momentum,
        masked_emb_var_weight=args.masked_emb_var_weight,
        lambda_mask=args.lambda_mask,
        n_mask_samples=args.n_mask_samples,
        view_noise_std=args.view_noise_std,
        triangle_alpha=args.triangle_alpha,
        confu_fuse_weight=args.confu_fuse_weight,
        use_cross_modal_tf=bool(args.use_cross_modal_tf),
        cross_modal_n_layers=args.cross_modal_n_layers,
        cross_modal_n_heads=args.cross_modal_n_heads,
        cross_modal_d_ff=args.cross_modal_d_ff,
    )
