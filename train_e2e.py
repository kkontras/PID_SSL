"""
End-to-end training: contrastive loss + classification cross-entropy jointly.
Implements Rung 8 of the experimental plan.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset_v3 import V3Config, V3DatasetGenerator, ALL_CONFIGS
from models.model_v3 import ModelV3Config, MultimodalContrastiveModel
from losses.combined import combined_loss


def _save_e2e_artifacts(history: List[Dict], save_dir: str) -> None:
    csv_path = os.path.join(save_dir, "e2e_history.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train_loss")
    axes[0].set_title("E2E train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot([row["epoch"] for row in history], [row["val_acc"] for row in history], label="val_acc", color="tab:green")
    axes[1].set_title("E2E val accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "e2e_curves.png"), dpi=140)
    plt.close(fig)


def train_e2e(
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
    classifier_hidden: int = 0,
    n_patches: int = 4,
    temperature: float = 0.07,
    lambda_contr: float = 0.1,
    batch_size: int = 512,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    target_mode: str = "composite",
    target_atom: Optional[str] = None,
    sigma_info: float = 0.01,
    mu_bg: float = 0.75,
    sigma_bg: float = 0.3,
    seed: int = 42,
    device: str = "cpu",
    save_dir: Optional[str] = None,
    log_every: int = 10,
) -> List[Dict]:
    """Run joint contrastive + supervised training."""
    print("=" * 72)
    print(f"E2E train | method={method} | config={config_name} | device={device} | seed={seed}")
    print("=" * 72)
    torch.manual_seed(seed)
    np.random.seed(seed)

    atoms = ALL_CONFIGS[config_name]
    data_cfg = V3Config(
        Q=Q,
        D=D,
        D_info=D_info,
        active_atoms=atoms,
        sigma_info=sigma_info,
        mu_bg=mu_bg,
        sigma_bg=sigma_bg,
        n_samples=n_train + n_val,
        seed=seed,
    )
    gen = V3DatasetGenerator(data_cfg)

    print(
        f"Generating dataset: train={n_train} val={n_val} Q={Q} D={D} D_info={D_info} "
        f"target_mode={target_mode} target_atom={target_atom} "
        f"sigma_info={sigma_info} mu_bg={mu_bg} sigma_bg={sigma_bg}"
    )
    data = gen.generate(n_train + n_val)
    x1_all = torch.from_numpy(data["x1"]).float()
    x2_all = torch.from_numpy(data["x2"]).float()
    x3_all = torch.from_numpy(data["x3"]).float()
    if target_mode == "composite":
        y_all = torch.from_numpy(data["label"]).long()
        n_classes = data_cfg.n_classes()
    elif target_mode == "sub_label":
        if target_atom is None:
            if len(atoms) != 1:
                raise ValueError("target_atom is required for sub_label mode when multiple atoms are active")
            target_atom = atoms[0]
        sub_key = f"sub_{target_atom}"
        if sub_key not in data:
            raise ValueError(f"Missing sub-label key: {sub_key}")
        y_all = torch.from_numpy(data[sub_key]).long()
        n_classes = Q
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    x1_tr, x1_va = x1_all[:n_train], x1_all[n_train:]
    x2_tr, x2_va = x2_all[:n_train], x2_all[n_train:]
    x3_tr, x3_va = x3_all[:n_train], x3_all[n_train:]
    y_tr, y_va = y_all[:n_train], y_all[n_train:]

    if classifier_hidden <= 0 and method == "none" and (
        config_name in {"A8", "A9", "A10", "A11"} or target_mode == "sub_label"
    ):
        classifier_hidden = 128

    model_cfg = ModelV3Config(
        d_input=D, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_model * 2, n_patches=n_patches, d_z=d_z, proj_hidden=d_model * 2,
        classifier_hidden=classifier_hidden,
        n_classes=n_classes,
    )
    model = MultimodalContrastiveModel(model_cfg).to(device)

    use_external_head = method == "none" and target_mode == "sub_label"
    supervised_head: Optional[nn.Module] = None
    if use_external_head:
        supervised_head = nn.Sequential(
            nn.Linear(3 * d_z, 2 * d_z),
            nn.ReLU(),
            nn.Linear(2 * d_z, n_classes),
        ).to(device)
        opt = torch.optim.AdamW(
            list(model.parameters()) + list(supervised_head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    n_steps = (n_train + batch_size - 1) // batch_size
    print(
        f"Model/training: batch_size={batch_size} epochs={epochs} "
        f"steps_per_epoch={n_steps} lr={lr} classifier_hidden={classifier_hidden} "
        f"use_external_head={use_external_head}"
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
            y_b = y_tr[idx].to(device)

            out = model.forward(x1_b, x2_b, x3_b)
            z1, z2, z3 = out["z1"], out["z2"], out["z3"]
            if supervised_head is not None:
                logits = supervised_head(torch.cat([z1, z2, z3], dim=-1))
            else:
                logits = out["logits"]

            fused = model.fuse(z1, z2, z3) if method == "confu" else None

            # SimCLR augmented views
            z1_a = z1_b_s = z2_a = z2_b_s = z3_a = z3_b_s = None
            if method == "simclr":
                noise_std = 0.1
                with torch.no_grad():
                    x1a = x1_b + noise_std * torch.randn_like(x1_b)
                    x1b = x1_b + noise_std * torch.randn_like(x1_b)
                    x2a = x2_b + noise_std * torch.randn_like(x2_b)
                    x2b = x2_b + noise_std * torch.randn_like(x2_b)
                    x3a = x3_b + noise_std * torch.randn_like(x3_b)
                    x3b = x3_b + noise_std * torch.randn_like(x3_b)
                oa = model.forward(x1a, x2a, x3a)
                ob = model.forward(x1b, x2b, x3b)
                z1_a, z2_a, z3_a = oa["z1"], oa["z2"], oa["z3"]
                z1_b_s, z2_b_s, z3_b_s = ob["z1"], ob["z2"], ob["z3"]

            loss, metrics = combined_loss(
                method=method, z1=z1, z2=z2, z3=z3,
                logits=logits, targets=y_b,
                fused=fused,
                z1_a=z1_a, z1_b=z1_b_s, z2_a=z2_a, z2_b=z2_b_s, z3_a=z3_a, z3_b=z3_b_s,
                temperature=temperature, lambda_contr=lambda_contr,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))

        if scheduler is not None:
            scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model.forward(x1_va.to(device), x2_va.to(device), x3_va.to(device))
            if supervised_head is not None:
                val_logits = supervised_head(torch.cat([val_out["z1"], val_out["z2"], val_out["z3"]], dim=-1))
            else:
                val_logits = val_out["logits"]
            val_acc = (val_logits.argmax(dim=-1) == y_va.to(device)).float().mean().item()

        row = {"epoch": epoch, "train_loss": np.mean(epoch_losses), "val_acc": val_acc}
        if epoch % log_every == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | train_loss={row['train_loss']:.4f} val_acc={val_acc:.3f}")
        history.append(row)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        payload = {"epoch": epochs, "model_state": model.state_dict(),
                    "model_cfg": model_cfg, "method": method, "config_name": config_name,
                    "target_mode": target_mode, "target_atom": target_atom,
                    "sigma_info": sigma_info, "mu_bg": mu_bg, "sigma_bg": sigma_bg}
        if supervised_head is not None:
            payload["supervised_head_state"] = supervised_head.state_dict()
        torch.save(payload, os.path.join(save_dir, "e2e_final.pt"))
        with open(os.path.join(save_dir, "e2e_history.json"), "w") as f:
            json.dump(history, f)
        _save_e2e_artifacts(history, save_dir)
        print(f"Saved e2e artifacts to: {save_dir}")

    best_val = max(row["val_acc"] for row in history)
    final_row = history[-1]
    print(
        f"Completed e2e | final_train_loss={final_row['train_loss']:.4f} "
        f"| final_val_acc={final_row['val_acc']:.3f} | best_val_acc={best_val:.3f}"
    )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="pairwise_nce",
                        choices=["none", "simclr", "pairwise_nce", "triangle", "confu"])
    parser.add_argument("--config", default="B4")
    parser.add_argument("--Q", type=int, default=7)
    parser.add_argument("--D", type=int, default=44)
    parser.add_argument("--D_info", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--lambda_contr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target_mode", choices=["composite", "sub_label"], default="composite")
    parser.add_argument("--target_atom", default=None)
    parser.add_argument("--sigma_info", type=float, default=0.01)
    parser.add_argument("--mu_bg", type=float, default=0.75)
    parser.add_argument("--sigma_bg", type=float, default=0.3)
    parser.add_argument("--classifier_hidden", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    train_e2e(
        method=args.method, config_name=args.config,
        Q=args.Q, D=args.D, D_info=args.D_info, n_train=args.n_train,
        classifier_hidden=args.classifier_hidden,
        target_mode=args.target_mode, target_atom=args.target_atom,
        sigma_info=args.sigma_info, mu_bg=args.mu_bg, sigma_bg=args.sigma_bg,
        lambda_contr=args.lambda_contr, temperature=args.tau,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        device=args.device, save_dir=args.save_dir,
    )
