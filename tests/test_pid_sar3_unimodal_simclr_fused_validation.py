from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator
from pid_sar3_ssl import (
    SSLEncoderConfig,
    SSLTrainConfig,
    UnimodalSimCLRModel,
    VectorAugmentationConfig,
    VectorAugmenter,
    encode_unimodal_numpy,
    family_from_pid_ids,
    train_unimodal_simclr,
)


PLOT_DIR = Path("test_outputs/pid_sar3_ssl_unimodal_fused")


def _ensure_plot_dir() -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return PLOT_DIR


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def _balanced_batch(gen: PIDSar3DatasetGenerator, n_per_pid: int, shuffle_seed: int, return_aux: bool = False) -> Dict[str, np.ndarray]:
    pid_ids = np.repeat(np.arange(10, dtype=np.int64), n_per_pid)
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(pid_ids)
    return gen.generate(n=int(pid_ids.size), pid_ids=pid_ids.tolist(), return_aux=return_aux)


def _fit_logreg_acc(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1200, random_state=0, multi_class="auto")
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    return float(accuracy_score(y_test, pred))


def _fit_ridge_r2(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, alpha: float = 1.0) -> float:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    reg = Ridge(alpha=alpha, random_state=0)
    reg.fit(Xtr, y_train)
    pred = reg.predict(Xte)
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2)) + 1e-8
    return float(1.0 - ss_res / ss_tot)


def _concat_raw(batch: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([batch["x1"], batch["x2"], batch["x3"]], axis=1).astype(np.float32)


def _concat_frozen_reps(models: Dict[str, UnimodalSimCLRModel], batch: Dict[str, np.ndarray]) -> np.ndarray:
    h1 = encode_unimodal_numpy(models["x1"], batch["x1"], device="cpu")
    h2 = encode_unimodal_numpy(models["x2"], batch["x2"], device="cpu")
    h3 = encode_unimodal_numpy(models["x3"], batch["x3"], device="cpu")
    return np.concatenate([h1, h2, h3], axis=1).astype(np.float32)


def _evaluate_supervised_suite(
    Xtr: np.ndarray,
    train_batch: Dict[str, np.ndarray],
    Xte: np.ndarray,
    test_batch: Dict[str, np.ndarray],
) -> Dict[str, float]:
    ytr_pid = train_batch["pid_id"].astype(np.int64)
    yte_pid = test_batch["pid_id"].astype(np.int64)
    ytr_fam = family_from_pid_ids(ytr_pid)
    yte_fam = family_from_pid_ids(yte_pid)

    out = {
        "pid10_acc": _fit_logreg_acc(Xtr, ytr_pid, Xte, yte_pid),
        "family3_acc": _fit_logreg_acc(Xtr, ytr_fam, Xte, yte_fam),
    }

    scalar_tasks = [
        ("y_u1", "mask_y_u1"),
        ("y_r12", "mask_y_r12"),
        ("y_r123", "mask_y_r123"),
        ("y_s12_3", "mask_y_s12_3"),
    ]
    for y_key, m_key in scalar_tasks:
        mtr = train_batch[m_key].astype(bool)
        mte = test_batch[m_key].astype(bool)
        out[f"n_train_{y_key}"] = float(np.sum(mtr))
        out[f"n_test_{y_key}"] = float(np.sum(mte))
        out[f"{y_key}_r2"] = _fit_ridge_r2(Xtr[mtr], train_batch[y_key][mtr], Xte[mte], test_batch[y_key][mte], alpha=1.0)
    return out


def test_plot_unimodal_simclr_frozen_fusion_supervised_validation():
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.45,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=1001,
        deleakage_fit_samples=1024,
    )
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)

    train_probe_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 1002}))
    test_probe_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 1003}))
    probe_train = _balanced_batch(train_probe_gen, n_per_pid=320, shuffle_seed=21, return_aux=True)
    probe_test = _balanced_batch(test_probe_gen, n_per_pid=140, shuffle_seed=22, return_aux=True)

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
        seed=23,
    )
    aug = VectorAugmenter(VectorAugmentationConfig(jitter_std=0.08, feature_drop_prob=0.08, gain_min=0.92, gain_max=1.08))

    models: Dict[str, UnimodalSimCLRModel] = {}
    train_rows: List[Dict[str, float]] = []
    for modality in ("x1", "x2", "x3"):
        model = UnimodalSimCLRModel(enc_cfg)
        hist = train_unimodal_simclr(model, ssl_gen, modality, train_cfg, augmenter=aug)
        models[modality] = model
        for row in hist:
            train_rows.append({"modality": modality, **row})

    Xtr_raw = _concat_raw(probe_train)
    Xte_raw = _concat_raw(probe_test)
    Xtr_fused = _concat_frozen_reps(models, probe_train)
    Xte_fused = _concat_frozen_reps(models, probe_test)

    res_raw = _evaluate_supervised_suite(Xtr_raw, probe_train, Xte_raw, probe_test)
    res_fused = _evaluate_supervised_suite(Xtr_fused, probe_train, Xte_fused, probe_test)

    task_order = ["pid10_acc", "family3_acc", "y_u1_r2", "y_r12_r2", "y_r123_r2", "y_s12_3_r2"]
    task_labels = ["PID-10 acc", "Family-3 acc", "R2(y_u1)", "R2(y_r12)", "R2(y_r123)", "R2(y_s12_3)"]
    rows = []
    for t in task_order:
        rows.append({"task": t, "label": task_labels[task_order.index(t)], "raw": float(res_raw[t]), "fused_simclr": float(res_fused[t]), "gain": float(res_fused[t] - res_raw[t])})

    # Figure 1: training curves
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for modality in ("x1", "x2", "x3"):
        rr = [r for r in train_rows if r["modality"] == modality]
        ax.plot([r["step"] for r in rr], [r["loss"] for r in rr], label=f"{modality} SimCLR", linewidth=2.0)
    ax.set_title("Unimodal SimCLR pretraining losses (frozen-fusion validation run)")
    ax.set_xlabel("step")
    ax.set_ylabel("NT-Xent loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _savefig(out_dir / "unimodal_simclr_fused_training_losses.png")

    # Figure 2: all supervised tasks summary
    x = np.arange(len(task_order))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    raw_vals = [r["raw"] for r in rows]
    fused_vals = [r["fused_simclr"] for r in rows]
    ax.bar(x - w / 2, raw_vals, width=w, label="Raw concat + linear probe", color="#9e9e9e")
    ax.bar(x + w / 2, fused_vals, width=w, label="Frozen unimodal SimCLR encoders (concat h) + linear probe", color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=18, ha="right")
    ax.set_ylabel("score (acc or R²)")
    ax.set_title("All supervised tasks with fused frozen encoders (concatenated modalities)")
    ax.grid(axis="y", alpha=0.25)
    for i, (rv, fv) in enumerate(zip(raw_vals, fused_vals)):
        ax.text(i - w / 2, rv + 0.015, f"{rv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, fv + 0.015, f"{fv:.2f}", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.25)
    _savefig(out_dir / "unimodal_simclr_fused_supervised_tasks_summary.png")

    # Figure 3: gains heatmap (single-row, crisp read)
    gains = np.array([[r["gain"] for r in rows]], dtype=np.float32)
    vmax = float(max(0.05, np.max(np.abs(gains))))
    fig, ax = plt.subplots(figsize=(12.4, 2.5))
    im = ax.imshow(gains, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(task_labels)))
    ax.set_xticklabels(task_labels, rotation=18, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels(["SimCLR fusion - raw"])
    ax.set_title("Task-wise gain from frozen unimodal SimCLR fusion")
    for j, v in enumerate(gains[0]):
        ax.text(j, 0, f"{v:+.2f}", ha="center", va="center", color="black" if abs(v) < 0.5 * vmax else "white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.subplots_adjust(bottom=0.34)
    _savefig(out_dir / "unimodal_simclr_fused_supervised_task_gains.png")

    # CSVs
    with (out_dir / "unimodal_simclr_fused_supervised_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "label", "raw", "fused_simclr", "gain"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with (out_dir / "unimodal_simclr_fused_supervised_metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k in ["n_train_y_u1", "n_test_y_u1", "n_train_y_r12", "n_test_y_r12", "n_train_y_r123", "n_test_y_r123", "n_train_y_s12_3", "n_test_y_s12_3"]:
            writer.writerow([k, int(res_fused[k])])

    with (out_dir / "unimodal_simclr_fused_training_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["modality", "step", "loss"])
        writer.writeheader()
        for r in train_rows:
            writer.writerow(r)

    assert np.all(np.isfinite([r["raw"] for r in rows]))
    assert np.all(np.isfinite([r["fused_simclr"] for r in rows]))
