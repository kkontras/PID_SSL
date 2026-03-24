from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset_v3 import V3Config, V3DatasetGenerator
from models.model_v3 import ModelV3Config, MultimodalContrastiveModel
from train_e2e import train_e2e


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_a8_split(
    Q: int,
    D: int,
    D_info: int,
    n_train: int,
    n_val: int,
    seed: int,
    sigma_info: float,
    mu_bg: float,
    sigma_bg: float,
) -> Dict[str, np.ndarray]:
    cfg = V3Config(
        Q=Q,
        D=D,
        D_info=D_info,
        active_atoms=["syn_12"],
        sigma_info=sigma_info,
        mu_bg=mu_bg,
        sigma_bg=sigma_bg,
        n_samples=n_train + n_val,
        seed=seed,
    )
    gen = V3DatasetGenerator(cfg)
    data = gen.generate(n_train + n_val)
    split: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        split[f"{key}_train"] = value[:n_train]
        split[f"{key}_val"] = value[n_train:]
    return split


def _save_history(history: List[Dict[str, float]], save_dir: Path, stem: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{stem}_history.csv"
    json_path = save_dir / f"{stem}_history.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    axes[0].set_title(f"{stem} loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train_acc")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val_acc")
    axes[1].set_title(f"{stem} accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_dir / f"{stem}_curves.png", dpi=140)
    plt.close(fig)


def _normalize(train: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (val - mean) / std


def run_raw_oracle(
    split: Dict[str, np.ndarray],
    save_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    device: str,
    seed: int,
) -> Dict[str, float]:
    _set_seed(seed)
    x_tr = np.concatenate([split["x1_train"], split["x2_train"], split["x3_train"]], axis=1)
    x_va = np.concatenate([split["x1_val"], split["x2_val"], split["x3_val"]], axis=1)
    x_tr, x_va = _normalize(x_tr, x_va)
    y_tr = split["sub_syn_12_train"]
    y_va = split["sub_syn_12_val"]

    x_tr_t = torch.from_numpy(x_tr).float().to(device)
    x_va_t = torch.from_numpy(x_va).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    y_va_t = torch.from_numpy(y_va).long().to(device)

    model = nn.Sequential(
        nn.Linear(x_tr.shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 7),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    rng = np.random.default_rng(seed)
    n_train = x_tr.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(x_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x_tr_t)
            val_logits = model(x_va_t)
            train_loss = F.cross_entropy(train_logits, y_tr_t).item()
            val_loss = F.cross_entropy(val_logits, y_va_t).item()
            train_acc = (train_logits.argmax(dim=-1) == y_tr_t).float().mean().item()
            val_acc = (val_logits.argmax(dim=-1) == y_va_t).float().mean().item()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )

    _save_history(history, save_dir, "raw_oracle")
    best_val = max(row["val_acc"] for row in history)
    return {"best_val_acc": best_val, "final_val_acc": history[-1]["val_acc"]}


def run_encoder_synergy(
    split: Dict[str, np.ndarray],
    save_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    device: str,
    seed: int,
    D: int,
) -> Dict[str, float]:
    _set_seed(seed)

    x1_tr = torch.from_numpy(split["x1_train"]).float()
    x2_tr = torch.from_numpy(split["x2_train"]).float()
    x3_tr = torch.from_numpy(split["x3_train"]).float()
    y_tr = torch.from_numpy(split["sub_syn_12_train"]).long()
    x1_va = torch.from_numpy(split["x1_val"]).float()
    x2_va = torch.from_numpy(split["x2_val"]).float()
    x3_va = torch.from_numpy(split["x3_val"]).float()
    y_va = torch.from_numpy(split["sub_syn_12_val"]).long()

    model_cfg = ModelV3Config(d_input=D, d_model=64, n_heads=4, n_layers=2, d_ff=128, n_patches=4, d_z=64, proj_hidden=128, n_classes=7)
    encoder = MultimodalContrastiveModel(model_cfg).to(device)
    head = nn.Sequential(
        nn.Linear(3 * model_cfg.d_z, 2 * model_cfg.d_z),
        nn.ReLU(),
        nn.Linear(2 * model_cfg.d_z, 7),
    ).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []
    rng = np.random.default_rng(seed)
    n_train = x1_tr.shape[0]

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx_np = perm[start:start + batch_size]
            idx = torch.from_numpy(idx_np).long()
            xb1 = x1_tr[idx].to(device)
            xb2 = x2_tr[idx].to(device)
            xb3 = x3_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            out = encoder.forward(xb1, xb2, xb3)
            features = torch.cat([out["z1"], out["z2"], out["z3"]], dim=-1)
            logits = head(features)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        encoder.eval()
        head.eval()
        with torch.no_grad():
            tr_out = encoder.forward(x1_tr.to(device), x2_tr.to(device), x3_tr.to(device))
            tr_logits = head(torch.cat([tr_out["z1"], tr_out["z2"], tr_out["z3"]], dim=-1))
            va_out = encoder.forward(x1_va.to(device), x2_va.to(device), x3_va.to(device))
            va_logits = head(torch.cat([va_out["z1"], va_out["z2"], va_out["z3"]], dim=-1))
            train_loss = F.cross_entropy(tr_logits, y_tr.to(device)).item()
            val_loss = F.cross_entropy(va_logits, y_va.to(device)).item()
            train_acc = (tr_logits.argmax(dim=-1) == y_tr.to(device)).float().mean().item()
            val_acc = (va_logits.argmax(dim=-1) == y_va.to(device)).float().mean().item()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "config": "A8",
            "target": "sub_syn_12",
        },
        save_dir / "encoder_synergy_final.pt",
    )
    _save_history(history, save_dir, "encoder_synergy")
    best_val = max(row["val_acc"] for row in history)
    return {"best_val_acc": best_val, "final_val_acc": history[-1]["val_acc"]}


def run_composite_e2e(
    save_dir: Path,
    Q: int,
    D: int,
    D_info: int,
    n_train: int,
    n_val: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> Dict[str, float]:
    history = train_e2e(
        method="none",
        config_name="A8",
        Q=Q,
        D=D,
        D_info=D_info,
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lambda_contr=0.0,
        seed=seed,
        device=device,
        save_dir=str(save_dir),
        log_every=max(1, epochs // 10),
    )
    best_val = max(row["val_acc"] for row in history)
    return {"best_val_acc": best_val, "final_val_acc": history[-1]["val_acc"]}


def interpret(raw_best: float, encoder_best: float, composite_best: float, chance: float) -> str:
    margin = 0.10
    raw_good = raw_best >= chance + margin
    enc_good = encoder_best >= chance + margin
    comp_good = composite_best >= chance + margin
    if not raw_good:
        return "dataset_or_target_issue"
    if raw_good and not enc_good and not comp_good:
        return "encoder_supervised_path_issue"
    if raw_good and enc_good and not comp_good:
        return "composite_label_benchmark_mismatch"
    if raw_good and enc_good and comp_good:
        return "all_solvable"
    return "mixed_or_inconclusive"


def save_summary(save_dir: Path, summary: Dict[str, object]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (save_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Q", type=int, default=7)
    parser.add_argument("--D", type=int, default=44)
    parser.add_argument("--D_info", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sigma_info", type=float, default=0.01)
    parser.add_argument("--mu_bg", type=float, default=0.75)
    parser.add_argument("--sigma_bg", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_root", default="test_outputs/a8_supervised_debug")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    print("=" * 72)
    print("A8 supervised debug")
    print("=" * 72)
    print(f"Output root: {out_root}")
    print(
        f"Dataset params: Q={args.Q} D={args.D} D_info={args.D_info} "
        f"sigma_info={args.sigma_info} mu_bg={args.mu_bg} sigma_bg={args.sigma_bg}"
    )
    split = _make_a8_split(
        args.Q,
        args.D,
        args.D_info,
        args.n_train,
        args.n_val,
        args.seed,
        args.sigma_info,
        args.mu_bg,
        args.sigma_bg,
    )

    raw_metrics = run_raw_oracle(
        split,
        out_root / "raw_oracle",
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
    )
    print(f"Raw oracle best_val_acc={raw_metrics['best_val_acc']:.3f}")

    encoder_metrics = run_encoder_synergy(
        split,
        out_root / "encoder_synergy",
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        D=args.D,
    )
    print(f"Encoder synergy best_val_acc={encoder_metrics['best_val_acc']:.3f}")

    composite_metrics = run_composite_e2e(
        out_root / "composite_e2e",
        Q=args.Q,
        D=args.D,
        D_info=args.D_info,
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    print(f"Composite e2e best_val_acc={composite_metrics['best_val_acc']:.3f}")

    chance = 1.0 / args.Q
    summary = {
        "config": "A8",
        "target": "sub_syn_12",
        "chance_acc": chance,
        "raw_oracle_best_val_acc": raw_metrics["best_val_acc"],
        "encoder_synergy_best_val_acc": encoder_metrics["best_val_acc"],
        "composite_e2e_best_val_acc": composite_metrics["best_val_acc"],
        "interpretation_tag": interpret(
            raw_metrics["best_val_acc"],
            encoder_metrics["best_val_acc"],
            composite_metrics["best_val_acc"],
            chance,
        ),
    }
    save_summary(out_root, summary)
    print(f"Interpretation: {summary['interpretation_tag']}")
    print(f"Saved summary to: {out_root}")


if __name__ == "__main__":
    main()
