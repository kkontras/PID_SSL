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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator, all_pid_names
from pid_sar3_ssl import (
    SSLEncoderConfig,
    SSLTrainConfig,
    UnimodalSimCLRModel,
    VectorAugmentationConfig,
    VectorAugmenter,
    encode_unimodal_numpy,
    train_unimodal_simclr,
)


PLOT_DIR = Path("test_outputs/pid_sar3_ssl_unimodal")
PID_NAMES = all_pid_names()


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


def _fit_probe_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=0, multi_class="auto")
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    cm = confusion_matrix(y_test, pred, labels=np.arange(10))
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {
        "acc": np.array([accuracy_score(y_test, pred)], dtype=np.float32),
        "confusion": cm.astype(np.int64),
        "per_class_recall": recall.astype(np.float32),
        "pred": pred.astype(np.int64),
    }


def _plot_confusions(confusions: Dict[str, np.ndarray], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.4))
    keys = ["x1", "x2", "x3"]
    for ax, key in zip(axes, keys):
        cm = confusions[key].astype(np.float32)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(cmn, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(f"{key} SimCLR probe confusion\n(row-normalized)")
        ax.set_xlabel("Predicted PID term")
        ax.set_ylabel("True PID term")
        ax.set_xticks(range(10))
        ax.set_xticklabels(PID_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(10))
        ax.set_yticklabels(PID_NAMES, fontsize=8)
        for i in range(10):
            ax.text(i, i, f"{cmn[i, i]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    _savefig(out_dir / "unimodal_simclr_confusions.png")


def test_plot_unimodal_simclr_pid_term_validation():
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.45,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=980,
        deleakage_fit_samples=1024,
    )
    train_gen = PIDSar3DatasetGenerator(data_cfg)
    probe_train_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 981}))
    probe_test_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 982}))

    probe_train = _balanced_batch(probe_train_gen, n_per_pid=260, shuffle_seed=11)
    probe_test = _balanced_batch(probe_test_gen, n_per_pid=120, shuffle_seed=12)
    y_train = probe_train["pid_id"].astype(np.int64)
    y_test = probe_test["pid_id"].astype(np.int64)

    enc_cfg = SSLEncoderConfig(
        input_dim=data_cfg.d,
        encoder_hidden_dim=96,
        representation_dim=48,
        projector_hidden_dim=96,
        projector_dim=48,
    )
    train_cfg = SSLTrainConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=192,
        steps=140,
        temperature=0.2,
        device="cpu",
        seed=17,
    )
    aug = VectorAugmenter(VectorAugmentationConfig(jitter_std=0.08, feature_drop_prob=0.08, gain_min=0.92, gain_max=1.08))

    summary_rows: List[Dict[str, float]] = []
    train_rows: List[Dict[str, float]] = []
    recall_rows: List[Dict[str, float]] = []
    confusion_simclr: Dict[str, np.ndarray] = {}

    simclr_recalls = np.zeros((3, 10), dtype=np.float32)
    raw_recalls = np.zeros((3, 10), dtype=np.float32)
    modality_order = ["x1", "x2", "x3"]

    for m_idx, modality in enumerate(modality_order):
        # Raw baseline probe
        raw_eval = _fit_probe_and_eval(probe_train[modality], y_train, probe_test[modality], y_test)
        raw_acc = float(raw_eval["acc"][0])
        raw_recalls[m_idx] = raw_eval["per_class_recall"]

        # Unimodal SimCLR training and frozen probe
        model = UnimodalSimCLRModel(enc_cfg)
        history = train_unimodal_simclr(model, train_gen, modality, train_cfg, augmenter=aug)
        for row in history:
            train_rows.append({"modality": modality, **row})

        Htr = encode_unimodal_numpy(model, probe_train[modality], device="cpu")
        Hte = encode_unimodal_numpy(model, probe_test[modality], device="cpu")
        simclr_eval = _fit_probe_and_eval(Htr, y_train, Hte, y_test)
        simclr_acc = float(simclr_eval["acc"][0])
        simclr_recalls[m_idx] = simclr_eval["per_class_recall"]
        confusion_simclr[modality] = simclr_eval["confusion"]

        summary_rows.append(
            {
                "modality": modality,
                "raw_probe_acc": raw_acc,
                "simclr_probe_acc": simclr_acc,
                "acc_gain": simclr_acc - raw_acc,
                "raw_macro_recall": float(np.mean(raw_eval["per_class_recall"])),
                "simclr_macro_recall": float(np.mean(simclr_eval["per_class_recall"])),
                "macro_recall_gain": float(np.mean(simclr_eval["per_class_recall"]) - np.mean(raw_eval["per_class_recall"])),
                "final_ssl_loss": float(history[-1]["loss"]),
                "mean_last10_ssl_loss": float(np.mean([r["loss"] for r in history[-10:]])),
            }
        )

        for pid_id, pid_name in enumerate(PID_NAMES):
            recall_rows.append(
                {
                    "modality": modality,
                    "pid_id": pid_id,
                    "pid_name": pid_name,
                    "raw_recall": float(raw_eval["per_class_recall"][pid_id]),
                    "simclr_recall": float(simclr_eval["per_class_recall"][pid_id]),
                    "recall_gain": float(simclr_eval["per_class_recall"][pid_id] - raw_eval["per_class_recall"][pid_id]),
                }
            )

    # Figure 1: SSL training curves per modality
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for modality in modality_order:
        rows = [r for r in train_rows if r["modality"] == modality]
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], linewidth=2.0, label=f"{modality} SimCLR")
    ax.set_title("Unimodal SimCLR training loss per modality")
    ax.set_xlabel("step")
    ax.set_ylabel("NT-Xent loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _savefig(out_dir / "unimodal_simclr_training_losses.png")

    # Figure 2: overall and macro recall bars
    x = np.arange(len(modality_order))
    w = 0.18
    raw_accs = [r["raw_probe_acc"] for r in summary_rows]
    sim_accs = [r["simclr_probe_acc"] for r in summary_rows]
    raw_mr = [r["raw_macro_recall"] for r in summary_rows]
    sim_mr = [r["simclr_macro_recall"] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - 1.5 * w, raw_accs, width=w, label="Raw acc", color="#9e9e9e")
    ax.bar(x - 0.5 * w, sim_accs, width=w, label="SimCLR acc", color="#4c78a8")
    ax.bar(x + 0.5 * w, raw_mr, width=w, label="Raw macro recall", color="#c7c7c7")
    ax.bar(x + 1.5 * w, sim_mr, width=w, label="SimCLR macro recall", color="#54a24b")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(modality_order)
    ax.set_ylabel("score")
    ax.set_title("Unimodal PID-term probe performance (10-way, held-out)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    for i, row in enumerate(summary_rows):
        ax.text(x[i] - 0.5 * w, row["simclr_probe_acc"] + 0.02, f"{row['simclr_probe_acc']:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + 1.5 * w, row["simclr_macro_recall"] + 0.02, f"{row['simclr_macro_recall']:.2f}", ha="center", fontsize=8)
    _savefig(out_dir / "unimodal_simclr_probe_summary.png")

    # Figure 3: per-class recall heatmap (raw vs simclr for each modality)
    heat = np.vstack([raw_recalls[0], simclr_recalls[0], raw_recalls[1], simclr_recalls[1], raw_recalls[2], simclr_recalls[2]])
    row_labels = ["x1 raw", "x1 simclr", "x2 raw", "x2 simclr", "x3 raw", "x3 simclr"]
    fig, ax = plt.subplots(figsize=(15.5, 5.5))
    im = ax.imshow(heat, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
    ax.set_xticks(range(10))
    ax.set_xticklabels(PID_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Per-PID-term recall (10-way linear probe): which terms are learned by each modality")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _savefig(out_dir / "unimodal_simclr_per_pid_recall_heatmap.png")

    # Figure 4: gain heatmap (SimCLR - raw)
    gain = np.vstack([simclr_recalls[0] - raw_recalls[0], simclr_recalls[1] - raw_recalls[1], simclr_recalls[2] - raw_recalls[2]])
    fig, ax = plt.subplots(figsize=(15.5, 3.8))
    vmax = float(max(0.05, np.max(np.abs(gain))))
    im = ax.imshow(gain, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(10))
    ax.set_xticklabels(PID_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(modality_order)
    ax.set_title("Per-PID-term recall gain from unimodal SimCLR (SimCLR - raw)")
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            ax.text(j, i, f"{gain[i, j]:+.2f}", ha="center", va="center", color="black" if abs(gain[i, j]) < vmax * 0.55 else "white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _savefig(out_dir / "unimodal_simclr_per_pid_recall_gain_heatmap.png")

    # Figure 5: row-normalized confusions for the SimCLR probe
    _plot_confusions(confusion_simclr, out_dir)

    # Save CSVs
    with (out_dir / "unimodal_simclr_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "modality",
                "raw_probe_acc",
                "simclr_probe_acc",
                "acc_gain",
                "raw_macro_recall",
                "simclr_macro_recall",
                "macro_recall_gain",
                "final_ssl_loss",
                "mean_last10_ssl_loss",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with (out_dir / "unimodal_simclr_per_pid_recall.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["modality", "pid_id", "pid_name", "raw_recall", "simclr_recall", "recall_gain"],
        )
        writer.writeheader()
        for row in recall_rows:
            writer.writerow(row)

    with (out_dir / "unimodal_simclr_training_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["modality", "step", "loss"])
        writer.writeheader()
        for row in train_rows:
            writer.writerow(row)

    with (out_dir / "unimodal_simclr_confusions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["modality", "true_pid", "pred_pid", "count"])
        for modality in modality_order:
            cm = confusion_simclr[modality]
            for i in range(10):
                for j in range(10):
                    writer.writerow([modality, PID_NAMES[i], PID_NAMES[j], int(cm[i, j])])

    assert len(summary_rows) == 3
    assert np.all(np.isfinite([r["simclr_probe_acc"] for r in summary_rows]))
    assert np.all(np.isfinite([r["simclr_macro_recall"] for r in summary_rows]))
