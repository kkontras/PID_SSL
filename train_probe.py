"""
Freeze pretrained encoder and run linear probe evaluation.
Implements Rung 7B of the experimental plan.
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

from data.dataset_v3 import V3Config, V3DatasetGenerator, ALL_CONFIGS
from models.model_v3 import ModelV3Config, MultimodalContrastiveModel
from probing.linear_probe import train_linear_probe
from probing.per_atom_eval import probe_per_atom
from probing.retrieval import evaluate_all_atom_retrieval


def _load_checkpoint(path: str, device: str) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_model_state_compat(model: MultimodalContrastiveModel, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load as much of a checkpoint as possible while tolerating older/newer model
    variants. Frozen probing only depends on the encoder/projection path, so
    missing unrelated heads should not block evaluation.
    """
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"Checkpoint compatibility: skipped {len(skipped)} mismatched keys")
    if missing:
        print(f"Checkpoint compatibility: missing {len(missing)} model keys after partial load")
    if unexpected:
        print(f"Checkpoint compatibility: unexpected {len(unexpected)} checkpoint keys ignored")


def _save_probe_artifacts(results: Dict, probe_histories: Dict[str, List[Dict[str, float]]], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "probe_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(save_dir, "probe_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe_key", "val_acc"])
        for probe_key, acc in results["overall"].items():
            writer.writerow([probe_key, acc])

    retrieval_rows = results.get("retrieval", [])
    if retrieval_rows:
        with open(os.path.join(save_dir, "retrieval_summary.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(retrieval_rows[0].keys()))
            writer.writeheader()
            writer.writerows(retrieval_rows)

    if probe_histories:
        with open(os.path.join(save_dir, "probe_histories.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["probe_key", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
            for probe_key, rows in probe_histories.items():
                for row in rows:
                    writer.writerow([probe_key, row["epoch"], row["train_loss"], row["val_loss"], row["train_acc"], row["val_acc"]])

        plt.figure(figsize=(8, 5))
        for probe_key, rows in probe_histories.items():
            epochs = [row["epoch"] for row in rows]
            plt.plot(epochs, [row["train_loss"] for row in rows], label=f"{probe_key} train")
            plt.plot(epochs, [row["val_loss"] for row in rows], linestyle="--", label=f"{probe_key} val")
        plt.xlabel("epoch")
        plt.ylabel("probe loss")
        plt.title("Frozen probe train/validation loss curves")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "probe_loss_curves.png"), dpi=140)
        plt.close()

        plt.figure(figsize=(8, 5))
        for probe_key, rows in probe_histories.items():
            epochs = [row["epoch"] for row in rows]
            plt.plot(epochs, [row["train_acc"] for row in rows], label=f"{probe_key} train")
            plt.plot(epochs, [row["val_acc"] for row in rows], linestyle="--", label=f"{probe_key} val")
        plt.xlabel("epoch")
        plt.ylabel("probe accuracy")
        plt.title("Frozen probe train/validation accuracy curves")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "probe_accuracy_curves.png"), dpi=140)
        plt.close()


@torch.no_grad()
def extract_representations(
    model: MultimodalContrastiveModel,
    x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
    device: str,
    batch_size: int = 512,
    use_confu: bool = False,
) -> Dict[str, np.ndarray]:
    """Extract frozen representations in batches.

    When the model has a CrossModalTransformer (use_cross_modal_tf=True), also
    extracts cross-modal CLS representations by passing full (unmasked) inputs
    through the joint Transformer and projecting the per-modality CLS outputs.
    These appear as cm_z1/cm_z2/cm_z3 in the returned dict.
    """
    model.eval()
    N = x1.shape[0]
    all_z1, all_z2, all_z3 = [], [], []
    all_f12, all_f13, all_f23 = [], [], []
    all_cm_z1, all_cm_z2, all_cm_z3 = [], [], []
    use_cross_modal_tf = model.cross_modal_tf is not None

    for start in range(0, N, batch_size):
        x1_b = x1[start:start + batch_size].to(device)
        x2_b = x2[start:start + batch_size].to(device)
        x3_b = x3[start:start + batch_size].to(device)
        out = model.forward(x1_b, x2_b, x3_b)
        all_z1.append(out["z1"].cpu().numpy())
        all_z2.append(out["z2"].cpu().numpy())
        all_z3.append(out["z3"].cpu().numpy())
        if use_confu:
            fused = model.fuse(out["z1"], out["z2"], out["z3"])
            all_f12.append(fused["f12"].cpu().numpy())
            all_f13.append(fused["f13"].cpu().numpy())
            all_f23.append(fused["f23"].cpu().numpy())
        if use_cross_modal_tf:
            cm_h1, cm_h2, cm_h3 = model.cross_modal_encode(x1_b, x2_b, x3_b)
            cm_z1, cm_z2, cm_z3 = model.project(cm_h1, cm_h2, cm_h3)
            all_cm_z1.append(cm_z1.cpu().numpy())
            all_cm_z2.append(cm_z2.cpu().numpy())
            all_cm_z3.append(cm_z3.cpu().numpy())

    result = {
        "z1": np.concatenate(all_z1),
        "z2": np.concatenate(all_z2),
        "z3": np.concatenate(all_z3),
    }
    if use_confu:
        result["f12"] = np.concatenate(all_f12)
        result["f13"] = np.concatenate(all_f13)
        result["f23"] = np.concatenate(all_f23)
    if use_cross_modal_tf:
        result["cm_z1"] = np.concatenate(all_cm_z1)
        result["cm_z2"] = np.concatenate(all_cm_z2)
        result["cm_z3"] = np.concatenate(all_cm_z3)
    return result


def run_probe(
    checkpoint_path: str,
    probe_config_name: str,
    Q: int = 7,
    D: int = 44,
    D_info: int = 4,
    n_probe_train: int = 5000,
    n_probe_test: int = 1000,
    probe_epochs: int = 100,
    device: str = "cpu",
    seed: int = 0,
    save_dir: Optional[str] = None,
) -> Dict:
    """Load a pretrained checkpoint and run linear probes."""
    print("=" * 72)
    print(f"Frozen probe | config={probe_config_name} | checkpoint={checkpoint_path} | device={device}")
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

    print(f"Generating probe dataset: train={n_probe_train} val={n_probe_test} Q={Q} D={D}")
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

    n_classes = data_cfg.n_classes()
    probe_key = "z1_z2_z3"
    probe_X_tr = np.concatenate([reps_tr["z1"], reps_tr["z2"], reps_tr["z3"]], axis=-1)
    probe_X_te = np.concatenate([reps_te["z1"], reps_te["z2"], reps_te["z3"]], axis=-1)
    print(f"Probe input: {probe_key} (concatenated all modalities)")

    res = train_linear_probe(
        probe_X_tr,
        y_tr,
        probe_X_te,
        y_te,
        n_classes=n_classes,
        epochs=probe_epochs,
        device=device,
        seed=seed,
    )
    results = {probe_key: res["val_acc"]}
    probe_histories: Dict[str, List[Dict[str, float]]] = {probe_key: res["history"]}
    print(f"  probe={probe_key:<10} val_acc={res['val_acc']:.3f}")

    # Cross-modal TF probe: representations from joint Transformer on full inputs
    if "cm_z1" in reps_tr:
        cm_probe_key = "cm_z1_cm_z2_cm_z3"
        cm_X_tr = np.concatenate([reps_tr["cm_z1"], reps_tr["cm_z2"], reps_tr["cm_z3"]], axis=-1)
        cm_X_te = np.concatenate([reps_te["cm_z1"], reps_te["cm_z2"], reps_te["cm_z3"]], axis=-1)
        cm_res = train_linear_probe(
            cm_X_tr, y_tr, cm_X_te, y_te,
            n_classes=n_classes, epochs=probe_epochs, device=device, seed=seed,
        )
        results[cm_probe_key] = cm_res["val_acc"]
        probe_histories[cm_probe_key] = cm_res["history"]
        print(f"  probe={cm_probe_key:<10} val_acc={cm_res['val_acc']:.3f}")

    # Per-atom probing
    sub_train = {f"sub_{a}": data[f"sub_{a}"][:n_probe_train] for a in atoms}
    sub_test = {f"sub_{a}": data[f"sub_{a}"][n_probe_train:] for a in atoms}
    per_atom = probe_per_atom(
        probe_X_tr,
        probe_X_te,
        sub_train,
        sub_test,
        Q=Q,
        atom_names=atoms,
        epochs=probe_epochs,
        device=device,
    )
    print(f"Per-atom probe on {probe_key}:")
    for atom_name, acc in per_atom.items():
        print(f"  atom={atom_name:<12} val_acc={acc:.3f}")

    per_atom_cm: Dict = {}
    if "cm_z1" in reps_tr:
        per_atom_cm = probe_per_atom(
            cm_X_tr,
            cm_X_te,
            sub_train,
            sub_test,
            Q=Q,
            atom_names=atoms,
            epochs=probe_epochs,
            device=device,
        )
        print(f"Per-atom probe on cm_z1_cm_z2_cm_z3:")
        for atom_name, acc in per_atom_cm.items():
            print(f"  atom={atom_name:<12} val_acc={acc:.3f}")

    atom_labels_train = {atom: data[f"sub_{atom}"][:n_probe_train] for atom in atoms}
    atom_labels_test = {atom: data[f"sub_{atom}"][n_probe_train:] for atom in atoms}
    retrieval_rows = evaluate_all_atom_retrieval(
        reps_train=reps_tr,
        reps_test=reps_te,
        atom_labels_train=atom_labels_train,
        atom_labels_test=atom_labels_test,
        method=method,
        device=device,
    )
    print("Retrieval summary (mean R@1 over atoms by query/mode):")
    by_query: Dict[str, List[float]] = {}
    for row in retrieval_rows:
        key = f"{row['query']}|{row['mode']}"
        by_query.setdefault(key, []).append(float(row["r_at_1"]))
    for key, vals in by_query.items():
        query, mode = key.split("|", 1)
        print(f"  query={query:<10} mode={mode:<14} mean_r1={np.mean(vals):.3f}")

    out = {
        "overall": results,
        "per_atom_all_modalities": per_atom,
        "per_atom_cm": per_atom_cm,
        "retrieval": retrieval_rows,
        "method": method,
        "config": probe_config_name,
    }
    if save_dir:
        _save_probe_artifacts(out, probe_histories, save_dir)
        print(f"Saved probe artifacts to: {save_dir}")
    print(f"Completed probe | input={probe_key} | val_acc={results[probe_key]:.3f}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe_config", default="A4")
    parser.add_argument("--Q", type=int, default=7)
    parser.add_argument("--D", type=int, default=44)
    parser.add_argument("--n_probe_train", type=int, default=5000)
    parser.add_argument("--n_probe_test", type=int, default=1000)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()

    results = run_probe(
        checkpoint_path=args.checkpoint,
        probe_config_name=args.probe_config,
        Q=args.Q, D=args.D,
        n_probe_train=args.n_probe_train, n_probe_test=args.n_probe_test,
        probe_epochs=args.probe_epochs, device=args.device, save_dir=args.save_dir,
    )
