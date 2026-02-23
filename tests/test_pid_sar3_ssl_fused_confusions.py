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


def _prepare_geometry_space(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize features and l2-normalize rows for cosine-geometry diagnostics."""
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    Xtr_n = Xtr_s / (np.linalg.norm(Xtr_s, axis=1, keepdims=True) + 1e-8)
    Xte_n = Xte_s / (np.linalg.norm(Xte_s, axis=1, keepdims=True) + 1e-8)
    return Xtr_n.astype(np.float32), Xte_n.astype(np.float32)


def _class_centroids(X: np.ndarray, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cents = []
    for lab in labels:
        Xi = X[y == lab]
        c = Xi.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        cents.append(c)
    return np.stack(cents, axis=0).astype(np.float32)


def _centroid_cosine_matrix(centroids: np.ndarray) -> np.ndarray:
    return (centroids @ centroids.T).astype(np.float32)


def _class_margin_stats(X: np.ndarray, y: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    label_to_idx = {int(l): i for i, l in enumerate(labels)}
    sims = X @ centroids.T  # cosine since both normalized
    margins = np.zeros((X.shape[0],), dtype=np.float32)
    class_means = np.zeros((labels.shape[0],), dtype=np.float32)
    for n in range(X.shape[0]):
        ci = label_to_idx[int(y[n])]
        true_sim = sims[n, ci]
        other_max = np.max(np.concatenate([sims[n, :ci], sims[n, ci + 1 :]]))
        margins[n] = float(true_sim - other_max)
    for i, lab in enumerate(labels):
        m = y == lab
        class_means[i] = float(np.mean(margins[m])) if np.any(m) else np.nan
    return class_means, float(np.mean(margins))


def _nearest_centroid_pair_acc(X: np.ndarray, y: np.ndarray, centroids: np.ndarray, a: int, b: int) -> float:
    mask = (y == a) | (y == b)
    Xp = X[mask]
    yp = y[mask]
    if Xp.shape[0] == 0:
        return float("nan")
    sims = Xp @ centroids[[a, b]].T
    pred = np.where(np.argmax(sims, axis=1) == 0, a, b)
    return float(np.mean(pred == yp))


def _compute_geometry_diagnostics(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> Dict[str, np.ndarray]:
    labels = np.arange(10, dtype=np.int64)
    Xtr_g, Xte_g = _prepare_geometry_space(Xtr, Xte)
    centroids = _class_centroids(Xtr_g, ytr, labels)
    C = _centroid_cosine_matrix(centroids)
    class_margin_means, overall_margin = _class_margin_stats(Xte_g, yte, centroids, labels)

    rs_pairs = [(3, 7), (4, 8), (5, 9)]  # (R12,S12->3), (R13,S13->2), (R23,S23->1)
    pair_accs = []
    pair_centroid_cos = []
    for a, b in rs_pairs:
        pair_accs.append(_nearest_centroid_pair_acc(Xte_g, yte, centroids, a, b))
        pair_centroid_cos.append(float(C[a, b]))

    family_groups = {
        "U": np.array([0, 1, 2], dtype=np.int64),
        "R": np.array([3, 4, 5, 6], dtype=np.int64),
        "S": np.array([7, 8, 9], dtype=np.int64),
    }
    family_margins = np.array([np.mean(class_margin_means[idxs]) for idxs in family_groups.values()], dtype=np.float32)

    return {
        "centroid_cosine": C,
        "class_margin_means": class_margin_means,
        "overall_margin": np.array([overall_margin], dtype=np.float32),
        "rs_pair_accs": np.asarray(pair_accs, dtype=np.float32),
        "rs_pair_centroid_cos": np.asarray(pair_centroid_cos, dtype=np.float32),
        "family_margin_means": family_margins,
    }


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


def _train_trimodal_objective(
    gen: PIDSar3DatasetGenerator,
    enc_cfg: SSLEncoderConfig,
    train_cfg: SSLTrainConfig,
    objective: str,
    model_name: str,
) -> Tuple[TriModalSSLModel, List[Dict[str, float]]]:
    model = TriModalSSLModel(enc_cfg)
    cfg = SSLTrainConfig(**{**train_cfg.__dict__, "objective": objective})
    hist = train_ssl(model, gen, cfg)
    rows = [{"model": model_name, "stream": "joint", **r} for r in hist]
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


def _plot_geometry_diagnostics(diag_a: Dict[str, np.ndarray], diag_b: Dict[str, np.ndarray], out_dir: Path) -> None:
    # Figure A: centroid cosine heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 6.4))
    for ax, D, title in zip(
        axes,
        [diag_a["centroid_cosine"], diag_b["centroid_cosine"]],
        ["Model A centroid cosine (train centroids)", "Model B centroid cosine (train centroids)"],
    ):
        im = ax.imshow(D, vmin=-0.3, vmax=1.0, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(10))
        ax.set_xticklabels(PID_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(10))
        ax.set_yticklabels(PID_NAMES, fontsize=8)
        ax.set_title(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Geometry diagnostic: PID class-centroid cosine structure", y=0.98)
    _savefig(out_dir / "geometry_pid_centroid_cosine_heatmaps.png")

    # Figure B: matched R<->S pair overlap + separability and family margins
    pair_labels = ["R12 vs S12->3", "R13 vs S13->2", "R23 vs S23->1"]
    x = np.arange(3)
    w = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 5.2))

    axes[0].bar(x - w / 2, diag_a["rs_pair_centroid_cos"], width=w, color="#4c78a8", label="Model A")
    axes[0].bar(x + w / 2, diag_b["rs_pair_centroid_cos"], width=w, color="#f58518", label="Model B")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pair_labels, rotation=22, ha="right")
    axes[0].set_ylabel("centroid cosine (higher = more overlap)")
    axes[0].set_title("Matched R/S pair centroid overlap")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x - w / 2, diag_a["rs_pair_accs"], width=w, color="#4c78a8")
    axes[1].bar(x + w / 2, diag_b["rs_pair_accs"], width=w, color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_labels, rotation=22, ha="right")
    axes[1].set_ylim(0.45, 1.02)
    axes[1].set_ylabel("nearest-centroid pair accuracy")
    axes[1].set_title("Matched R/S pair separability (held-out)")
    axes[1].grid(axis="y", alpha=0.25)

    fam_labels = ["U", "R", "S"]
    xf = np.arange(3)
    axes[2].bar(xf - w / 2, diag_a["family_margin_means"], width=w, color="#4c78a8")
    axes[2].bar(xf + w / 2, diag_b["family_margin_means"], width=w, color="#f58518")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[2].set_xticks(xf)
    axes[2].set_xticklabels(fam_labels)
    axes[2].set_ylabel("mean class margin")
    axes[2].set_title("Family-level margin geometry")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Geometry diagnostics in fused frozen representation space", y=0.98)
    fig.subplots_adjust(bottom=0.28, wspace=0.35)
    _savefig(out_dir / "geometry_rs_overlap_and_margins.png")


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

    # Geometry diagnostics on fused frozen representation spaces (train centroids, test evaluation).
    ytr_pid = probe_train["pid_id"].astype(np.int64)
    yte_pid = probe_test["pid_id"].astype(np.int64)
    geom_a = _compute_geometry_diagnostics(Xtr_a, ytr_pid, Xte_a, yte_pid)
    geom_b = _compute_geometry_diagnostics(Xtr_b, ytr_pid, Xte_b, yte_pid)

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
    _plot_geometry_diagnostics(geom_a, geom_b, out_dir)

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

    # Geometry CSVs (compact + class-level details)
    with (out_dir / "fused_frozen_two_models_geometry_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "item", "model_a_unimodal_simclr_sum", "model_b_pairwise_infonce_sum", "b_minus_a"])
        writer.writerow(
            [
                "overall_margin",
                "all_pid_classes",
                float(geom_a["overall_margin"][0]),
                float(geom_b["overall_margin"][0]),
                float(geom_b["overall_margin"][0] - geom_a["overall_margin"][0]),
            ]
        )
        for fam, i in zip(["U", "R", "S"], range(3)):
            writer.writerow(
                [
                    "family_mean_margin",
                    fam,
                    float(geom_a["family_margin_means"][i]),
                    float(geom_b["family_margin_means"][i]),
                    float(geom_b["family_margin_means"][i] - geom_a["family_margin_means"][i]),
                ]
            )
        for lbl, i in zip(["R12_vs_S12->3", "R13_vs_S13->2", "R23_vs_S23->1"], range(3)):
            writer.writerow(
                [
                    "matched_rs_pair_nearest_centroid_acc",
                    lbl,
                    float(geom_a["rs_pair_accs"][i]),
                    float(geom_b["rs_pair_accs"][i]),
                    float(geom_b["rs_pair_accs"][i] - geom_a["rs_pair_accs"][i]),
                ]
            )
            writer.writerow(
                [
                    "matched_rs_pair_centroid_cosine",
                    lbl,
                    float(geom_a["rs_pair_centroid_cos"][i]),
                    float(geom_b["rs_pair_centroid_cos"][i]),
                    float(geom_b["rs_pair_centroid_cos"][i] - geom_a["rs_pair_centroid_cos"][i]),
                ]
            )

    with (out_dir / "fused_frozen_two_models_class_margins.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pid_id", "pid_name", "model_a_class_margin", "model_b_class_margin", "b_minus_a"])
        for pid_id, pid_name in enumerate(PID_NAMES):
            a_val = float(geom_a["class_margin_means"][pid_id])
            b_val = float(geom_b["class_margin_means"][pid_id])
            writer.writerow([pid_id, pid_name, a_val, b_val, b_val - a_val])

    assert np.isfinite(eval_a["pid10_acc"]) and np.isfinite(eval_b["pid10_acc"])
    assert np.isfinite(eval_a["family3_acc"]) and np.isfinite(eval_b["family3_acc"])


def test_plot_fused_confusions_four_models_higher_order():
    """
    Four-way fused frozen-encoder comparison:
    - Model A: sum of 3 unimodal SimCLR streams
    - Model B: sum of 3 pairwise InfoNCE losses (pairwise NT-Xent; SimCLR-style)
    - Model C: TRIANGLE-like (pairwise NT-Xent + triad shape regularizer)
    - Model D: ConFu-style (pairwise NT-Xent + fused-pair-to-third contrastive terms)
    """
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.45,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=1401,
        deleakage_fit_samples=1024,
    )
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)
    probe_train_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 1402}))
    probe_test_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": 1403}))
    probe_train = _balanced_batch(probe_train_gen, n_per_pid=340, shuffle_seed=41, return_aux=True)
    probe_test = _balanced_batch(probe_test_gen, n_per_pid=160, shuffle_seed=42, return_aux=True)

    enc_cfg = SSLEncoderConfig(input_dim=data_cfg.d, encoder_hidden_dim=96, representation_dim=48, projector_hidden_dim=96, projector_dim=48)
    base_train_cfg = SSLTrainConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=192,
        steps=140,
        temperature=0.2,
        device="cpu",
        seed=51,
        triangle_reg_weight=0.15,
        confu_pair_weight=0.5,
        confu_fused_weight=0.5,
    )

    # Train all four models.
    unimodal_models, hist_a = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, base_train_cfg)
    Xtr_a = _concat_unimodal_frozen(unimodal_models, probe_train)
    Xte_a = _concat_unimodal_frozen(unimodal_models, probe_test)

    model_b, hist_b = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "pairwise_simclr", "sum_3_pairwise_infonce")
    Xtr_b = _concat_trimodal_frozen(model_b, probe_train)
    Xte_b = _concat_trimodal_frozen(model_b, probe_test)

    model_c, hist_c = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "triangle_like", "triangle_like")
    Xtr_c = _concat_trimodal_frozen(model_c, probe_train)
    Xte_c = _concat_trimodal_frozen(model_c, probe_test)

    model_d, hist_d = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "confu_style", "confu_style")
    Xtr_d = _concat_trimodal_frozen(model_d, probe_train)
    Xte_d = _concat_trimodal_frozen(model_d, probe_test)

    ytr_pid = probe_train["pid_id"].astype(np.int64)
    yte_pid = probe_test["pid_id"].astype(np.int64)

    evals = {
        "model_a_unimodal_simclr_sum": _evaluate_all_tasks(Xtr_a, probe_train, Xte_a, probe_test),
        "model_b_pairwise_infonce_sum": _evaluate_all_tasks(Xtr_b, probe_train, Xte_b, probe_test),
        "model_c_triangle_like": _evaluate_all_tasks(Xtr_c, probe_train, Xte_c, probe_test),
        "model_d_confu_style": _evaluate_all_tasks(Xtr_d, probe_train, Xte_d, probe_test),
    }
    geoms = {
        "model_a_unimodal_simclr_sum": _compute_geometry_diagnostics(Xtr_a, ytr_pid, Xte_a, yte_pid),
        "model_b_pairwise_infonce_sum": _compute_geometry_diagnostics(Xtr_b, ytr_pid, Xte_b, yte_pid),
        "model_c_triangle_like": _compute_geometry_diagnostics(Xtr_c, ytr_pid, Xte_c, yte_pid),
        "model_d_confu_style": _compute_geometry_diagnostics(Xtr_d, ytr_pid, Xte_d, yte_pid),
    }

    model_titles = {
        "model_a_unimodal_simclr_sum": "A: 3x unimodal SimCLR",
        "model_b_pairwise_infonce_sum": "B: pairwise InfoNCE sum\n(pairwise SimCLR/NT-Xent)",
        "model_c_triangle_like": "C: TRIANGLE-like",
        "model_d_confu_style": "D: ConFu-style",
    }

    # Primary figure: 2x2 PID-10 confusion matrices.
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 14.0))
    keys = list(model_titles.keys())
    for ax, key in zip(axes.flat, keys):
        cm = evals[key]["_pid_cm"]  # type: ignore[index]
        cmn = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(cmn, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(model_titles[key])
        ax.set_xlabel("Predicted PID term")
        ax.set_ylabel("True PID term")
        ax.set_xticks(range(10))
        ax.set_xticklabels(PID_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(10))
        ax.set_yticklabels(PID_NAMES, fontsize=8)
        # annotate diagonal
        for i in range(10):
            ax.text(i, i, f"{cmn[i, i]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
    fig.suptitle("PID-10 confusion matrices: fused frozen encoders + linear probe (4-model comparison)", y=0.99)
    _savefig(out_dir / "pid10_confusions_fused_frozen_four_models.png")

    # Geometry comparison (compact bar-panel)
    geo_keys = ["model_a_unimodal_simclr_sum", "model_b_pairwise_infonce_sum", "model_c_triangle_like", "model_d_confu_style"]
    geo_labels = ["A: 3x uni SimCLR", "B: pairwise InfoNCE", "C: TRIANGLE-like", "D: ConFu-style"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.4))

    # Mean matched R/S centroid cosine (lower is better)
    rs_cos_means = [float(np.mean(geoms[k]["rs_pair_centroid_cos"])) for k in geo_keys]
    axes[0].bar(x, rs_cos_means, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(geo_labels, rotation=20, ha="right")
    axes[0].set_ylabel("mean matched R/S centroid cosine")
    axes[0].set_title("Matched R/S centroid overlap\n(lower is better)")
    axes[0].grid(axis="y", alpha=0.25)

    # Mean matched-pair nearest-centroid accuracy (higher is better)
    rs_acc_means = [float(np.mean(geoms[k]["rs_pair_accs"])) for k in geo_keys]
    axes[1].bar(x, rs_acc_means, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(geo_labels, rotation=20, ha="right")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_ylabel("mean matched-pair NC accuracy")
    axes[1].set_title("Matched R/S separability\n(higher is better)")
    axes[1].grid(axis="y", alpha=0.25)

    # PID-10 accuracy (classification summary paired with confusion primary)
    pid_accs = [float(evals[k]["pid10_acc"]) for k in geo_keys]
    axes[2].bar(x, pid_accs, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(geo_labels, rotation=20, ha="right")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("PID-10 accuracy")
    axes[2].set_title("Classification summary\n(complements confusions)")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Geometry + PID summary (4-model fused frozen comparison)", y=0.99)
    fig.subplots_adjust(bottom=0.28, wspace=0.35)
    _savefig(out_dir / "geometry_pid_summary_four_models.png")

    # Secondary all-task summary (compact heatmap for 4 models x tasks)
    task_order = ["pid10_acc", "family3_acc", "y_u1_r2", "y_r12_r2", "y_r123_r2", "y_s12_3_r2"]
    task_labels = ["PID-10 acc", "Family-3 acc", "R2(y_u1)", "R2(y_r12)", "R2(y_r123)", "R2(y_s12_3)"]
    mat = np.array([[float(evals[k][t]) for t in task_order] for k in geo_keys], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(task_labels)))
    ax.set_xticklabels(task_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(geo_labels)))
    ax.set_yticklabels(geo_labels)
    ax.set_title("Secondary all-task summary (fused frozen encoders + linear probes)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.subplots_adjust(bottom=0.28)
    _savefig(out_dir / "all_tasks_heatmap_four_models.png")

    # CSV summaries
    with (out_dir / "fused_frozen_four_models_task_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *task_order])
        for k in geo_keys:
            writer.writerow([k] + [float(evals[k][t]) for t in task_order])

    with (out_dir / "fused_frozen_four_models_geometry_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "overall_margin",
                "family_margin_U",
                "family_margin_R",
                "family_margin_S",
                "mean_matched_rs_centroid_cos",
                "mean_matched_rs_nc_acc",
                "rs_cos_R12_S12",
                "rs_cos_R13_S13",
                "rs_cos_R23_S23",
                "rs_ncacc_R12_S12",
                "rs_ncacc_R13_S13",
                "rs_ncacc_R23_S23",
            ]
        )
        for k in geo_keys:
            g = geoms[k]
            writer.writerow(
                [
                    k,
                    float(g["overall_margin"][0]),
                    float(g["family_margin_means"][0]),
                    float(g["family_margin_means"][1]),
                    float(g["family_margin_means"][2]),
                    float(np.mean(g["rs_pair_centroid_cos"])),
                    float(np.mean(g["rs_pair_accs"])),
                    float(g["rs_pair_centroid_cos"][0]),
                    float(g["rs_pair_centroid_cos"][1]),
                    float(g["rs_pair_centroid_cos"][2]),
                    float(g["rs_pair_accs"][0]),
                    float(g["rs_pair_accs"][1]),
                    float(g["rs_pair_accs"][2]),
                ]
            )

    with (out_dir / "fused_frozen_four_models_confusions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "true_label", "pred_label", "count"])
        for k in geo_keys:
            cm = evals[k]["_pid_cm"]  # type: ignore[index]
            for i, li in enumerate(PID_NAMES):
                for j, lj in enumerate(PID_NAMES):
                    writer.writerow([k, li, lj, int(cm[i, j])])

    with (out_dir / "fused_frozen_four_models_training_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "stream", "step", "loss"])
        writer.writeheader()
        for r in hist_a + hist_b + hist_c + hist_d:
            writer.writerow({k: r.get(k, "") for k in ["model", "stream", "step", "loss"]})

    # Sanity checks
    for k in geo_keys:
        assert np.isfinite(float(evals[k]["pid10_acc"]))
