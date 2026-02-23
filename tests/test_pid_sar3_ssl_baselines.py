from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator
from pid_sar3_ssl import (
    SSLEncoderConfig,
    SSLTrainConfig,
    TriModalSSLModel,
    concat_representations,
    encode_numpy,
    family_from_pid_ids,
    train_ssl,
)


PLOT_DIR = Path("test_outputs/pid_sar3_ssl")


def _ensure_plot_dir() -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return PLOT_DIR


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def _balanced_batch(gen: PIDSar3DatasetGenerator, n_per_pid: int, shuffle_seed: int) -> Dict[str, np.ndarray]:
    pid_ids = np.repeat(np.arange(10, dtype=np.int64), n_per_pid)
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(pid_ids)
    return gen.generate(n=int(pid_ids.size), pid_ids=pid_ids.tolist())


def _logreg_accuracy(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=800, random_state=0, multi_class="auto")
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    return float(accuracy_score(y_test, pred))


def _pair_cosines(reps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for a, b in (("x1", "x2"), ("x1", "x3"), ("x2", "x3")):
        xa = reps[a]
        xb = reps[b]
        xa = xa / (np.linalg.norm(xa, axis=1, keepdims=True) + 1e-8)
        xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-8)
        out[f"{a}_{b}"] = np.sum(xa * xb, axis=1)
    return out


def _cosine_summary_by_family(reps: Dict[str, np.ndarray], pid_ids: np.ndarray) -> Dict[str, float]:
    fam = family_from_pid_ids(pid_ids)
    pair_cos = _pair_cosines(reps)
    rows: Dict[str, float] = {}
    family_names = {0: "unique", 1: "redundancy", 2: "synergy"}
    for fam_id, fam_name in family_names.items():
        mask = fam == fam_id
        if not np.any(mask):
            rows[f"{fam_name}_mean_cos"] = float("nan")
            continue
        vals = []
        for k in pair_cos:
            vals.append(float(np.mean(pair_cos[k][mask])))
        rows[f"{fam_name}_mean_cos"] = float(np.mean(vals))
    for k, v in pair_cos.items():
        rows[f"{k}_mean_cos"] = float(np.mean(v))
    rows["overall_mean_cross_modal_cos"] = float(
        np.mean([rows["x1_x2_mean_cos"], rows["x1_x3_mean_cos"], rows["x2_x3_mean_cos"]])
    )
    return rows


def _raw_as_representations(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {"x1": batch["x1"].copy(), "x2": batch["x2"].copy(), "x3": batch["x3"].copy()}


def test_plot_ssl_baseline_results():
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.45,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=900,
        deleakage_fit_samples=1024,
    )
    train_gen = PIDSar3DatasetGenerator(data_cfg)
    probe_train_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 901}))
    probe_test_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 902}))

    probe_train = _balanced_batch(probe_train_gen, n_per_pid=220, shuffle_seed=1)
    probe_test = _balanced_batch(probe_test_gen, n_per_pid=90, shuffle_seed=2)

    encoder_cfg = SSLEncoderConfig(
        input_dim=data_cfg.d,
        encoder_hidden_dim=96,
        representation_dim=48,
        projector_hidden_dim=96,
        projector_dim=48,
    )

    objective_specs = [
        ("pairwise_simclr", "Pairwise SimCLR (sum over pairs)"),
        ("tri_positive_infonce", "Tri-positive InfoNCE"),
    ]

    training_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    histories: Dict[str, List[Dict[str, float]]] = {}
    rep_cache_test: Dict[str, Dict[str, np.ndarray]] = {"raw_concat": _raw_as_representations(probe_test)}

    for obj_key, obj_label in objective_specs:
        model = TriModalSSLModel(encoder_cfg)
        train_cfg = SSLTrainConfig(
            objective=obj_key,
            lr=1e-3,
            weight_decay=1e-5,
            batch_size=192,
            steps=120,
            temperature=0.2,
            device="cpu",
            seed=7,
        )
        history = train_ssl(model=model, generator=train_gen, cfg=train_cfg)
        histories[obj_key] = history
        for row in history:
            training_rows.append({"objective": obj_key, **row})

        reps_train = encode_numpy(model, probe_train, device="cpu")
        reps_test = encode_numpy(model, probe_test, device="cpu")
        rep_cache_test[obj_key] = reps_test

        Xtr = concat_representations(reps_train)
        Xte = concat_representations(reps_test)
        ytr_pid = probe_train["pid_id"].astype(np.int64)
        yte_pid = probe_test["pid_id"].astype(np.int64)
        ytr_fam = family_from_pid_ids(ytr_pid)
        yte_fam = family_from_pid_ids(yte_pid)

        pid_acc = _logreg_accuracy(Xtr, ytr_pid, Xte, yte_pid)
        family_acc = _logreg_accuracy(Xtr, ytr_fam, Xte, yte_fam)
        cos_stats = _cosine_summary_by_family(reps_test, yte_pid)

        last10 = np.mean([r["loss"] for r in history[-10:]])
        summary_rows.append(
            {
                "objective": obj_key,
                "label": obj_label,
                "steps": float(train_cfg.steps),
                "final_loss": float(history[-1]["loss"]),
                "mean_last10_loss": float(last10),
                "pid_probe_acc": float(pid_acc),
                "family_probe_acc": float(family_acc),
                **cos_stats,
            }
        )

    # Raw baseline probe on concatenated observations for reference.
    raw_train = _raw_as_representations(probe_train)
    raw_test = rep_cache_test["raw_concat"]
    raw_pid_acc = _logreg_accuracy(concat_representations(raw_train), probe_train["pid_id"], concat_representations(raw_test), probe_test["pid_id"])
    raw_family_acc = _logreg_accuracy(
        concat_representations(raw_train),
        family_from_pid_ids(probe_train["pid_id"]),
        concat_representations(raw_test),
        family_from_pid_ids(probe_test["pid_id"]),
    )
    summary_rows.insert(
        0,
        {
            "objective": "raw_concat",
            "label": "Raw concat (no SSL)",
            "steps": 0.0,
            "final_loss": float("nan"),
            "mean_last10_loss": float("nan"),
            "pid_probe_acc": float(raw_pid_acc),
            "family_probe_acc": float(raw_family_acc),
            **_cosine_summary_by_family(raw_test, probe_test["pid_id"]),
        },
    )

    # Figure 1: training curves
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for obj_key, obj_label in objective_specs:
        hist = histories[obj_key]
        steps = [int(r["step"]) for r in hist]
        losses = [float(r["loss"]) for r in hist]
        ax.plot(steps, losses, linewidth=2.0, label=obj_label)
    ax.set_title("Tri-modal SSL training loss (x1, x2, x3 as modalities)")
    ax.set_xlabel("step")
    ax.set_ylabel("contrastive loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _savefig(out_dir / "ssl_training_loss_curves.png")

    # Figure 2: probe accuracies
    labels = [r["label"] for r in summary_rows]
    pid_vals = [r["pid_probe_acc"] for r in summary_rows]
    fam_vals = [r["family_probe_acc"] for r in summary_rows]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.bar(x - w / 2, pid_vals, width=w, label="10-way PID probe acc", color="#4c78a8")
    ax.bar(x + w / 2, fam_vals, width=w, label="3-way family probe acc", color="#54a24b")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title("Frozen-representation linear probe accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for i, (pv, fv) in enumerate(zip(pid_vals, fam_vals)):
        ax.text(i - w / 2, pv + 0.015, f"{pv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, fv + 0.015, f"{fv:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.24)
    _savefig(out_dir / "ssl_probe_accuracy_summary.png")

    # Figure 3: mean cross-modal cosine by atom family
    fam_keys = ["unique_mean_cos", "redundancy_mean_cos", "synergy_mean_cos"]
    fam_labels = ["Unique", "Redundancy", "Synergy"]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    x = np.arange(len(fam_keys))
    width = 0.22
    colors = ["#9e9e9e", "#f58518", "#4c78a8"]
    for idx, row in enumerate(summary_rows):
        vals = [row[k] for k in fam_keys]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=row["label"], color=colors[idx], alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(fam_labels)
    ax.set_ylabel("mean same-sample cosine across modality pairs")
    ax.set_title("Cross-modal alignment tendency by PID family (held-out)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _savefig(out_dir / "ssl_cross_modal_cosine_by_family.png")

    # Machine-readable outputs
    summary_csv = out_dir / "ssl_baseline_summary.csv"
    summary_fields = [
        "objective",
        "label",
        "steps",
        "final_loss",
        "mean_last10_loss",
        "pid_probe_acc",
        "family_probe_acc",
        "x1_x2_mean_cos",
        "x1_x3_mean_cos",
        "x2_x3_mean_cos",
        "overall_mean_cross_modal_cos",
        "unique_mean_cos",
        "redundancy_mean_cos",
        "synergy_mean_cos",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    train_csv = out_dir / "ssl_training_curves.csv"
    train_fields = ["objective", "step", "loss", "loss_12", "loss_13", "loss_23", "loss_anchor_x1", "loss_anchor_x2", "loss_anchor_x3"]
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=train_fields)
        writer.writeheader()
        for row in training_rows:
            out = {k: row.get(k, "") for k in train_fields}
            writer.writerow(out)

    assert len(summary_rows) == 3
    assert np.all(np.isfinite([r["pid_probe_acc"] for r in summary_rows]))
    assert np.all(np.isfinite([r["family_probe_acc"] for r in summary_rows]))
    assert np.isfinite(summary_rows[1]["mean_last10_loss"])
    assert np.isfinite(summary_rows[2]["mean_last10_loss"])
