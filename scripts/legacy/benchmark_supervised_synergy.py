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

from data.dataset_v3 import ALL_CONFIGS, V3Config, V3DatasetGenerator
from train_e2e import train_e2e


SYNERGY_CONFIGS = ["A8", "A9", "A10", "A11"]
TARGET_ATOM = {
    "A8": "syn_12",
    "A9": "syn_13",
    "A10": "syn_23",
    "A11": "syn_123",
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_split(
    config_name: str,
    Q: int,
    D: int,
    D_info: int,
    n_train: int,
    n_val: int,
    sigma_info: float,
    mu_bg: float,
    sigma_bg: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    cfg = V3Config(
        Q=Q,
        D=D,
        D_info=D_info,
        active_atoms=ALL_CONFIGS[config_name],
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


def normalize(train: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (val - mean) / std


def save_history(history: List[Dict[str, float]], save_dir: Path, stem: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / f"{stem}_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    axes[0].set_title(f"{stem} loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train_acc")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val_acc")
    axes[1].set_title(f"{stem} accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_dir / f"{stem}_curves.png", dpi=140)
    plt.close(fig)


def run_raw_oracle(
    split: Dict[str, np.ndarray],
    target_atom: str,
    Q: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
    save_dir: Path,
) -> Dict[str, float]:
    set_seed(seed)
    x_tr = np.concatenate([split["x1_train"], split["x2_train"], split["x3_train"]], axis=1)
    x_va = np.concatenate([split["x1_val"], split["x2_val"], split["x3_val"]], axis=1)
    x_tr, x_va = normalize(x_tr, x_va)
    y_tr = split[f"sub_{target_atom}_train"]
    y_va = split[f"sub_{target_atom}_val"]

    x_tr_t = torch.from_numpy(x_tr).float().to(device)
    x_va_t = torch.from_numpy(x_va).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    y_va_t = torch.from_numpy(y_va).long().to(device)

    model = nn.Sequential(
        nn.Linear(x_tr.shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, Q),
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
            tr_logits = model(x_tr_t)
            va_logits = model(x_va_t)
            train_loss = F.cross_entropy(tr_logits, y_tr_t).item()
            val_loss = F.cross_entropy(va_logits, y_va_t).item()
            train_acc = (tr_logits.argmax(dim=-1) == y_tr_t).float().mean().item()
            val_acc = (va_logits.argmax(dim=-1) == y_va_t).float().mean().item()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )
    save_history(history, save_dir, "raw_oracle")
    best = max(history, key=lambda row: row["val_acc"])
    return {"best_epoch": int(best["epoch"]), "best_val_acc": best["val_acc"], "final_val_acc": history[-1]["val_acc"]}


def run_encoder_sub_label(
    config_name: str,
    target_atom: str,
    Q: int,
    D: int,
    D_info: int,
    n_train: int,
    n_val: int,
    epochs: int,
    batch_size: int,
    lr: float,
    sigma_info: float,
    mu_bg: float,
    sigma_bg: float,
    device: str,
    seed: int,
    save_dir: Path,
) -> Dict[str, float]:
    history = train_e2e(
        method="none",
        config_name=config_name,
        Q=Q,
        D=D,
        D_info=D_info,
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lambda_contr=0.0,
        target_mode="sub_label",
        target_atom=target_atom,
        sigma_info=sigma_info,
        mu_bg=mu_bg,
        sigma_bg=sigma_bg,
        seed=seed,
        device=device,
        save_dir=str(save_dir),
        log_every=max(1, epochs // 10),
    )
    best = max(history, key=lambda row: row["val_acc"])
    return {"best_epoch": int(best["epoch"]), "best_val_acc": best["val_acc"], "final_val_acc": history[-1]["val_acc"]}


def run_composite(
    config_name: str,
    Q: int,
    D: int,
    D_info: int,
    n_train: int,
    n_val: int,
    epochs: int,
    batch_size: int,
    lr: float,
    sigma_info: float,
    mu_bg: float,
    sigma_bg: float,
    device: str,
    seed: int,
    save_dir: Path,
) -> Dict[str, float]:
    history = train_e2e(
        method="none",
        config_name=config_name,
        Q=Q,
        D=D,
        D_info=D_info,
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lambda_contr=0.0,
        target_mode="composite",
        sigma_info=sigma_info,
        mu_bg=mu_bg,
        sigma_bg=sigma_bg,
        seed=seed,
        device=device,
        save_dir=str(save_dir),
        log_every=max(1, epochs // 10),
    )
    best = max(history, key=lambda row: row["val_acc"])
    return {"best_epoch": int(best["epoch"]), "best_val_acc": best["val_acc"], "final_val_acc": history[-1]["val_acc"]}


def interpretation(raw_best: float, enc_best: float, comp_best: float, chance: float) -> str:
    margin = 0.10
    raw_good = raw_best >= chance + margin
    enc_good = enc_best >= chance + margin
    comp_good = comp_best >= chance + margin
    if enc_good and not comp_good:
        return "composite_label_benchmark_mismatch"
    if raw_good and enc_good and comp_good:
        return "all_solvable"
    if raw_good and not enc_good:
        return "encoder_supervised_path_issue"
    if not raw_good and enc_good:
        return "raw_oracle_underpowered_or_misconfigured"
    if not raw_good:
        return "dataset_or_target_issue"
    return "mixed_or_inconclusive"


def save_summary(rows: List[Dict[str, object]], out_root: Path) -> None:
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(rows))
    width = 0.24
    ax.bar(x - width, [r["raw_oracle_best_val_acc"] for r in rows], width=width, label="Raw Oracle")
    ax.bar(x, [r["encoder_sub_label_best_val_acc"] for r in rows], width=width, label="Encoder Sub-label")
    ax.bar(x + width, [r["composite_best_val_acc"] for r in rows], width=width, label="Composite")
    ax.set_xticks(x)
    ax.set_xticklabels([r["config"] for r in rows])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Supervised Synergy Benchmark")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "summary_barplot.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=SYNERGY_CONFIGS)
    parser.add_argument("--Q", type=int, default=7)
    parser.add_argument("--D", type=int, default=44)
    parser.add_argument("--D_info", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sigma_info", type=float, default=0.002)
    parser.add_argument("--mu_bg", type=float, default=0.40)
    parser.add_argument("--sigma_bg", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_root", default="test_outputs/supervised_synergy_benchmark")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for config_name in args.configs:
        target_atom = TARGET_ATOM[config_name]
        config_root = out_root / config_name
        print("=" * 72)
        print(f"Synergy benchmark | config={config_name} | target={target_atom}")
        print("=" * 72)

        split = make_split(
            config_name=config_name,
            Q=args.Q,
            D=args.D,
            D_info=args.D_info,
            n_train=args.n_train,
            n_val=args.n_val,
            sigma_info=args.sigma_info,
            mu_bg=args.mu_bg,
            sigma_bg=args.sigma_bg,
            seed=args.seed,
        )
        raw = run_raw_oracle(
            split=split,
            target_atom=target_atom,
            Q=args.Q,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
            save_dir=config_root / "raw_oracle",
        )
        encoder = run_encoder_sub_label(
            config_name=config_name,
            target_atom=target_atom,
            Q=args.Q,
            D=args.D,
            D_info=args.D_info,
            n_train=args.n_train,
            n_val=args.n_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigma_info=args.sigma_info,
            mu_bg=args.mu_bg,
            sigma_bg=args.sigma_bg,
            device=args.device,
            seed=args.seed,
            save_dir=config_root / "encoder_sub_label",
        )
        composite = run_composite(
            config_name=config_name,
            Q=args.Q,
            D=args.D,
            D_info=args.D_info,
            n_train=args.n_train,
            n_val=args.n_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigma_info=args.sigma_info,
            mu_bg=args.mu_bg,
            sigma_bg=args.sigma_bg,
            device=args.device,
            seed=args.seed,
            save_dir=config_root / "composite_e2e",
        )

        chance = 1.0 / args.Q
        row = {
            "config": config_name,
            "target_atom": target_atom,
            "chance_acc": chance,
            "raw_oracle_best_val_acc": raw["best_val_acc"],
            "encoder_sub_label_best_val_acc": encoder["best_val_acc"],
            "composite_best_val_acc": composite["best_val_acc"],
            "interpretation_tag": interpretation(raw["best_val_acc"], encoder["best_val_acc"], composite["best_val_acc"], chance),
            "sigma_info": args.sigma_info,
            "mu_bg": args.mu_bg,
            "sigma_bg": args.sigma_bg,
        }
        rows.append(row)
        with (config_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print(json.dumps(row, indent=2))

    save_summary(rows, out_root)


if __name__ == "__main__":
    main()
