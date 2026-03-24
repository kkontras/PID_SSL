from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset_v3 import ALL_CONFIGS, V3Config, V3DatasetGenerator
from models.model_v3 import MultimodalContrastiveModel
from probing.nonlinear_probe import train_nonlinear_probe
from train_probe import _load_checkpoint, _load_model_state_compat, extract_representations


def _save_artifacts(results: Dict, histories: Dict[str, List[Dict[str, float]]], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "nonlinear_probe_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(save_dir, "nonlinear_probe_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe_key", "val_acc"])
        for key, value in results["overall"].items():
            writer.writerow([key, value])
    with open(os.path.join(save_dir, "nonlinear_probe_histories.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe_key", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for probe_key, rows in histories.items():
            for row in rows:
                writer.writerow([probe_key, row["epoch"], row["train_loss"], row["val_loss"], row["train_acc"], row["val_acc"]])

    plt.figure(figsize=(8, 5))
    for probe_key, rows in histories.items():
        epochs = [row["epoch"] for row in rows]
        plt.plot(epochs, [row["train_acc"] for row in rows], label=f"{probe_key} train")
        plt.plot(epochs, [row["val_acc"] for row in rows], linestyle="--", label=f"{probe_key} val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Nonlinear frozen probe accuracy curves")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "nonlinear_probe_accuracy_curves.png"), dpi=140)
    plt.close()


def run_nonlinear_probe(
    checkpoint_path: str,
    probe_config_name: str,
    Q: int = 7,
    D: int = 44,
    D_info: int = 4,
    n_probe_train: int = 3000,
    n_probe_test: int = 1000,
    probe_epochs: int = 300,
    hidden_dim: int = 256,
    device: str = "cpu",
    seed: int = 0,
    save_dir: Optional[str] = None,
) -> Dict:
    print("=" * 72)
    print(f"Nonlinear frozen probe | config={probe_config_name} | checkpoint={checkpoint_path} | device={device}")
    print("=" * 72)
    ckpt = _load_checkpoint(checkpoint_path, device)
    model_cfg = ckpt["model_cfg"]
    method = ckpt.get("method", "unknown")

    model = MultimodalContrastiveModel(model_cfg).to(device)
    _load_model_state_compat(model, ckpt["model_state"])
    model.eval()

    atoms = ALL_CONFIGS[probe_config_name]
    data_cfg = V3Config(Q=Q, D=D, D_info=D_info, active_atoms=atoms, n_samples=n_probe_train + n_probe_test, seed=seed + 999)
    gen = V3DatasetGenerator(data_cfg)
    data = gen.generate(n_probe_train + n_probe_test)
    x1 = torch.from_numpy(data["x1"]).float()
    x2 = torch.from_numpy(data["x2"]).float()
    x3 = torch.from_numpy(data["x3"]).float()
    labels = data["label"]

    x1_tr, x1_te = x1[:n_probe_train], x1[n_probe_train:]
    x2_tr, x2_te = x2[:n_probe_train], x2[n_probe_train:]
    x3_tr, x3_te = x3[:n_probe_train], x3[n_probe_train:]
    y_tr, y_te = labels[:n_probe_train], labels[n_probe_train:]

    reps_tr = extract_representations(model, x1_tr, x2_tr, x3_tr, device, use_confu=(method == "confu"))
    reps_te = extract_representations(model, x1_te, x2_te, x3_te, device, use_confu=(method == "confu"))

    probe_key = "z1_z2_z3"
    X_tr = np.concatenate([reps_tr["z1"], reps_tr["z2"], reps_tr["z3"]], axis=-1)
    X_te = np.concatenate([reps_te["z1"], reps_te["z2"], reps_te["z3"]], axis=-1)
    result = train_nonlinear_probe(
        X_tr,
        y_tr,
        X_te,
        y_te,
        n_classes=data_cfg.n_classes(),
        epochs=probe_epochs,
        hidden_dim=hidden_dim,
        device=device,
        seed=seed,
    )
    out = {
        "overall": {probe_key: result["val_acc"]},
        "method": method,
        "config": probe_config_name,
        "probe_type": "nonlinear",
    }
    histories = {probe_key: result["history"]}
    if save_dir:
        _save_artifacts(out, histories, save_dir)
        print(f"Saved nonlinear probe artifacts to: {save_dir}")
    print(f"Completed nonlinear probe | input={probe_key} | val_acc={result['val_acc']:.3f}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe_config", required=True)
    parser.add_argument("--Q", type=int, default=7)
    parser.add_argument("--D", type=int, default=44)
    parser.add_argument("--D_info", type=int, default=4)
    parser.add_argument("--n_probe_train", type=int, default=3000)
    parser.add_argument("--n_probe_test", type=int, default=1000)
    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    run_nonlinear_probe(
        checkpoint_path=args.checkpoint,
        probe_config_name=args.probe_config,
        Q=args.Q,
        D=args.D,
        D_info=args.D_info,
        n_probe_train=args.n_probe_train,
        n_probe_test=args.n_probe_test,
        probe_epochs=args.probe_epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
        save_dir=args.save_dir,
    )
