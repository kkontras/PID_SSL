from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator, all_pid_names
from pid_sar3_ssl import (
    SSLEncoderConfig,
    SSLTrainConfig,
    TriModalSSLModel,
    UnimodalSimCLRModel,
    VectorAugmentationConfig,
    VectorAugmenter,
    encode_numpy,
    encode_unimodal_numpy,
    family_from_pid_ids,
    train_ssl,
    train_unimodal_simclr,
)


PLOT_DIR = Path("test_outputs/pid_sar3_ssl_fused_confusions")
PID_NAMES = all_pid_names()
FAMILY_NAMES = ["Unique", "Redundancy", "Synergy"]


def _ensure_plot_dir() -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return PLOT_DIR


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _balanced_batch(gen: PIDSar3DatasetGenerator, n_per_pid: int, shuffle_seed: int, return_aux: bool = False) -> Dict[str, np.ndarray]:
    pid_ids = np.repeat(np.arange(10, dtype=np.int64), n_per_pid)
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(pid_ids)
    return gen.generate(n=int(pid_ids.size), pid_ids=pid_ids.tolist(), return_aux=return_aux)


def _fit_classifier_with_confusion(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=1500, random_state=0, multi_class="auto")
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)
    cm = confusion_matrix(yte, pred, labels=labels)
    return {
        "acc": np.array([accuracy_score(yte, pred)], dtype=np.float32),
        "cm": cm.astype(np.int64),
        "pred": pred.astype(np.int64),
    }


def _fit_ridge_r2(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> float:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    reg = Ridge(alpha=1.0)
    reg.fit(Xtr_s, ytr)
    pred = reg.predict(Xte_s)
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - np.mean(yte)) ** 2)) + 1e-8
    return float(1.0 - ss_res / ss_tot)


def _concat_raw(batch: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([batch["x1"], batch["x2"], batch["x3"]], axis=1).astype(np.float32)


def _concat_unimodal_frozen(models: Dict[str, UnimodalSimCLRModel], batch: Dict[str, np.ndarray]) -> np.ndarray:
    h1 = encode_unimodal_numpy(models["x1"], batch["x1"], device="cpu")
    h2 = encode_unimodal_numpy(models["x2"], batch["x2"], device="cpu")
    h3 = encode_unimodal_numpy(models["x3"], batch["x3"], device="cpu")
    return np.concatenate([h1, h2, h3], axis=1).astype(np.float32)


def _concat_trimodal_frozen(model: TriModalSSLModel, batch: Dict[str, np.ndarray]) -> np.ndarray:
    reps = encode_numpy(model, batch, device="cpu")
    return np.concatenate([reps["x1"], reps["x2"], reps["x3"]], axis=1).astype(np.float32)


def _train_model_a_unimodal_sum_simclr(gen: PIDSar3DatasetGenerator, enc_cfg: SSLEncoderConfig, train_cfg: SSLTrainConfig) -> Tuple[Dict[str, UnimodalSimCLRModel], List[Dict[str, float]]]:
    aug = VectorAugmenter(VectorAugmentationConfig(jitter_std=0.08, feature_drop_prob=0.08, gain_min=0.92, gain_max=1.08))
    models: Dict[str, UnimodalSimCLRModel] = {}
    rows: List[Dict[str, float]] = []
    for modality in ("x1", "x2", "x3"):
        model = UnimodalSimCLRModel(enc_cfg)
        hist = train_unimodal_simclr(model, gen, modality, train_cfg, augmenter=aug)
        models[modality] = model
        for r in hist:
            rows.append({"model": "sum_3_unimodal_simclr", "stream": modality, **r})
    return models, rows


def _train_model_b_pairwise_infonce(gen: PIDSar3DatasetGenerator, enc_cfg: SSLEncoderConfig, train_cfg: SSLTrainConfig) -> Tuple[TriModalSSLModel, List[Dict[str, float]]]:
    model = TriModalSSLModel(enc_cfg)
    cfg = SSLTrainConfig(**{**train_cfg.__dict__, "objective": "pairwise_simclr"})
    hist = train_ssl(model, gen, cfg)
    rows = [{"model": "sum_3_pairwise_infonce", "stream": "joint", **r} for r in hist]
    return model, rows


def _evaluate_all_tasks(Xtr: np.ndarray, train_batch: Dict[str, np.ndarray], Xte: np.ndarray, test_batch: Dict[str, np.ndarray]) -> Dict[str, float]:
    ytr_pid = train_batch["pid_id"].astype(np.int64)
    yte_pid = test_batch["pid_id"].astype(np.int64)
    ytr_fam = family_from_pid_ids(ytr_pid)
    yte_fam = family_from_pid_ids(yte_pid)

    pid_eval = _fit_classifier_with_confusion(Xtr, ytr_pid, Xte, yte_pid, labels=np.arange(10))
    fam_eval = _fit_classifier_with_confusion(Xtr, ytr_fam, Xte, yte_fam, labels=np.arange(3))

    out: Dict[str, float] = {
        "pid10_acc": float(pid_eval["acc"][0]),
        "family3_acc": float(fam_eval["acc"][0]),
    }
    for key, mkey in [("y_u1", "mask_y_u1"), ("y_r12", "mask_y_r12"), ("y_r123", "mask_y_r123"), ("y_s12_3", "mask_y_s12_3")]:
        mtr = train_batch[mkey].astype(bool)
        mte = test_batch[mkey].astype(bool)
        out[f"{key}_r2"] = _fit_ridge_r2(Xtr[mtr], train_batch[key][mtr], Xte[mte], test_batch[key][mte])
        out[f"n_train_{key}"] = float(np.sum(mtr))
        out[f"n_test_{key}"] = float(np.sum(mte))
    out["_pid_cm"] = pid_eval["cm"]  # type: ignore[assignment]
    out["_family_cm"] = fam_eval["cm"]  # type: ignore[assignment]
    return out


def _plot_two_confusions(cm_a: np.ndarray, cm_b: np.ndarray, labels: List[str], title_a: str, title_b: str, out_path: Path, super_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 6.4))
    for ax, cm, title in zip(axes, [cm_a, cm_b], [title_a, title_b]):
        cmn = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(cmn, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            ax.text(i, i, f"{cmn[i, i]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(super_title, y=0.98)
    _savefig(out_path)


def test_plot_fused_confusions_two_models():
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.45,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=1201,
        deleakage_fit_samples=1024,
    )
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)
    probe_train_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 1202}))
    probe_test_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 1203}))
    probe_train = _balanced_batch(probe_train_gen, n_per_pid=360, shuffle_seed=31, return_aux=True)
    probe_test = _balanced_batch(probe_test_gen, n_per_pid=160, shuffle_seed=32, return_aux=True)

    enc_cfg = SSLEncoderConfig(input_dim=data_cfg.d, encoder_hidden_dim=96, representation_dim=48, projector_hidden_dim=96, projector_dim=48)
    base_train_cfg = SSLTrainConfig(lr=1e-3, weight_decay=1e-5, batch_size=192, steps=160, temperature=0.2, device="cpu", seed=41)

    # Model A: sum of 3 unimodal SimCLR losses (one SimCLR stream per modality)
    unimodal_models, hist_a = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, base_train_cfg)
    Xtr_a = _concat_unimodal_frozen(unimodal_models, probe_train)
    Xte_a = _concat_unimodal_frozen(unimodal_models, probe_test)
    eval_a = _evaluate_all_tasks(Xtr_a, probe_train, Xte_a, probe_test)

    # Model B: 3 cross-modal pairwise InfoNCE losses (x1-x2, x1-x3, x2-x3), summed
    trimodal_model, hist_b = _train_model_b_pairwise_infonce(ssl_gen, enc_cfg, base_train_cfg)
    Xtr_b = _concat_trimodal_frozen(trimodal_model, probe_train)
    Xte_b = _concat_trimodal_frozen(trimodal_model, probe_test)
    eval_b = _evaluate_all_tasks(Xtr_b, probe_train, Xte_b, probe_test)

    # Confusion matrices are the primary output.
    _plot_two_confusions(
        eval_a["_pid_cm"],  # type: ignore[index]
        eval_b["_pid_cm"],  # type: ignore[index]
        PID_NAMES,
        "Model A: sum of 3 unimodal SimCLR (frozen h concat)",
        "Model B: sum of 3 pairwise InfoNCE (frozen h concat)",
        out_dir / "pid10_confusions_fused_frozen_two_models.png",
        "PID-10 confusion matrices (all modalities concatenated, frozen encoders + linear probe)",
    )
    _plot_two_confusions(
        eval_a["_family_cm"],  # type: ignore[index]
        eval_b["_family_cm"],  # type: ignore[index]
        FAMILY_NAMES,
        "Model A: sum of 3 unimodal SimCLR",
        "Model B: sum of 3 pairwise InfoNCE",
        out_dir / "family3_confusions_fused_frozen_two_models.png",
        "Family-3 confusion matrices (all modalities concatenated, frozen encoders + linear probe)",
    )

    # Compact all-task summary (secondary to confusions).
    task_order = ["pid10_acc", "family3_acc", "y_u1_r2", "y_r12_r2", "y_r123_r2", "y_s12_3_r2"]
    task_labels = ["PID-10 acc", "Family-3 acc", "R2(y_u1)", "R2(y_r12)", "R2(y_r123)", "R2(y_s12_3)"]
    rows = []
    for t, lbl in zip(task_order, task_labels):
        rows.append(
            {
                "task": t,
                "label": lbl,
                "model_a_unimodal_simclr_sum": float(eval_a[t]),
                "model_b_pairwise_infonce_sum": float(eval_b[t]),
                "b_minus_a": float(eval_b[t] - eval_a[t]),
            }
        )

    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2, [r["model_a_unimodal_simclr_sum"] for r in rows], width=w, color="#4c78a8", label="Model A: sum of 3 unimodal SimCLR")
    ax.bar(x + w / 2, [r["model_b_pairwise_infonce_sum"] for r in rows], width=w, color="#f58518", label="Model B: sum of 3 pairwise InfoNCE")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=18, ha="right")
    ax.set_ylabel("score (acc or R²)")
    ax.set_title("Secondary summary: fused frozen-encoder linear probes on all supervised tasks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=0.26)
    _savefig(out_dir / "all_supervised_tasks_fused_frozen_two_models.png")

    # Training curves (supporting)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    for modality in ("x1", "x2", "x3"):
        rr = [r for r in hist_a if r["stream"] == modality]
        axes[0].plot([r["step"] for r in rr], [r["loss"] for r in rr], linewidth=1.8, label=modality)
    axes[0].set_title("Model A training losses (3 unimodal SimCLR streams)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot([r["step"] for r in hist_b], [r["loss"] for r in hist_b], linewidth=2.0, color="#f58518")
    axes[1].set_title("Model B training loss (sum of 3 pairwise InfoNCE)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss")
    axes[1].grid(alpha=0.25)
    _savefig(out_dir / "training_curves_two_models.png")

    # CSV outputs
    with (out_dir / "fused_frozen_two_models_task_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "label", "model_a_unimodal_simclr_sum", "model_b_pairwise_infonce_sum", "b_minus_a"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with (out_dir / "fused_frozen_two_models_training_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "stream", "step", "loss"])
        writer.writeheader()
        for r in hist_a + hist_b:
            writer.writerow({k: r.get(k, "") for k in ["model", "stream", "step", "loss"]})

    with (out_dir / "fused_frozen_two_models_confusions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "model", "true_label", "pred_label", "count"])
        for target, key, labels in [("pid10", "_pid_cm", PID_NAMES), ("family3", "_family_cm", FAMILY_NAMES)]:
            for model_name, ev in [("model_a_unimodal_simclr_sum", eval_a), ("model_b_pairwise_infonce_sum", eval_b)]:
                cm = ev[key]  # type: ignore[index]
                for i, li in enumerate(labels):
                    for j, lj in enumerate(labels):
                        writer.writerow([target, model_name, li, lj, int(cm[i, j])])

    assert np.isfinite(eval_a["pid10_acc"]) and np.isfinite(eval_b["pid10_acc"])
    assert np.isfinite(eval_a["family3_acc"]) and np.isfinite(eval_b["family3_acc"])
