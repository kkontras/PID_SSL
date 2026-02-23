from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
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


def _data_cfg_compositional_very_easy(seed: int) -> PIDDatasetConfig:
    return PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.02,
        rho_choices=(0.8,),
        hop_choices=(1,),
        seed=int(seed),
        deleakage_fit_samples=1024,
        composition_mode="multi_atom",
        active_atoms_per_sample=5,
        shared_backbone_gain=4.0,
        shared_backbone_tied_projection=True,
        synergy_deleak_lambda=0.25,
    )


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


def _fit_binary_macro_f1_kappa_over_target_dims(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    Yte: np.ndarray,
) -> Dict[str, float]:
    """
    Predict a target modality vector dimension-wise as median-thresholded binary tasks.
    Returns macro averages over target dimensions for F1 and Cohen's kappa.
    """
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    f1s: List[float] = []
    kappas: List[float] = []
    d = int(Ytr.shape[1])
    for j in range(d):
        thr = float(np.median(Ytr[:, j]))
        ytr = (Ytr[:, j] > thr).astype(np.int64)
        yte = (Yte[:, j] > thr).astype(np.int64)
        # Guard pathological constant-label folds (rare with continuous values, but possible).
        if int(np.unique(ytr).size) < 2:
            pred = np.full_like(yte, fill_value=int(ytr[0]))
        else:
            clf = LogisticRegression(max_iter=1000, random_state=0)
            clf.fit(Xtr_s, ytr)
            pred = clf.predict(Xte_s).astype(np.int64)
        f1s.append(float(f1_score(yte, pred, zero_division=0)))
        kappas.append(float(cohen_kappa_score(yte, pred)))
    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_kappa": float(np.mean(kappas)),
        "n_target_dims": float(d),
    }


def _fit_reconstruction_decoder_metrics(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    Yte: np.ndarray,
    decoder: str,
) -> Dict[str, float]:
    """
    Frozen-feature multi-output reconstruction benchmark on raw target modality vectors.

    Reports macro R^2 (mean over target dimensions) and macro normalized RMSE
    where each dimension is normalized by the train-split standard deviation.
    """
    x_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(Xtr)
    Xte_s = x_scaler.transform(Xte)

    if decoder == "ridge":
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr_s, Ytr)
        pred = reg.predict(Xte_s).astype(np.float32)
    elif decoder == "mlp":
        y_scaler = StandardScaler()
        Ytr_s = y_scaler.fit_transform(Ytr)
        reg = MLPRegressor(
            hidden_layer_sizes=(48,),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=60,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=6,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            reg.fit(Xtr_s, Ytr_s)
        pred = y_scaler.inverse_transform(reg.predict(Xte_s)).astype(np.float32)
    else:
        raise ValueError(f"Unknown decoder: {decoder}")

    r2s: List[float] = []
    nrmse: List[float] = []
    d = int(Ytr.shape[1])
    for j in range(d):
        ytr_j = Ytr[:, j].astype(np.float64)
        yte_j = Yte[:, j].astype(np.float64)
        pred_j = pred[:, j].astype(np.float64)
        r2s.append(float(r2_score(yte_j, pred_j)))
        scale = float(np.std(ytr_j)) + 1e-8
        rmse = float(np.sqrt(np.mean((yte_j - pred_j) ** 2)))
        nrmse.append(float(rmse / scale))
    return {
        "macro_r2": float(np.mean(r2s)),
        "macro_nrmse": float(np.mean(nrmse)),
        "n_target_dims": float(d),
    }


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


def _split_modalities_from_concat(X: np.ndarray) -> Dict[str, np.ndarray]:
    d = X.shape[1] // 3
    return {"x1": X[:, :d], "x2": X[:, d : 2 * d], "x3": X[:, 2 * d : 3 * d]}


def _train_model_a_unimodal_sum_simclr(
    gen: PIDSar3DatasetGenerator,
    enc_cfg: SSLEncoderConfig,
    train_cfg: SSLTrainConfig,
    pid_schedule: Tuple[int, ...] | None = None,
) -> Tuple[Dict[str, UnimodalSimCLRModel], List[Dict[str, float]]]:
    aug = VectorAugmenter(VectorAugmentationConfig(jitter_std=0.08, feature_drop_prob=0.08, gain_min=0.92, gain_max=1.08))
    models: Dict[str, UnimodalSimCLRModel] = {}
    rows: List[Dict[str, float]] = []
    for modality in ("x1", "x2", "x3"):
        model = UnimodalSimCLRModel(enc_cfg)
        hist = train_unimodal_simclr(model, gen, modality, train_cfg, augmenter=aug, pid_schedule=pid_schedule)
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
    pid_schedule: Tuple[int, ...] | None = None,
) -> Tuple[TriModalSSLModel, List[Dict[str, float]]]:
    model = TriModalSSLModel(enc_cfg)
    cfg = SSLTrainConfig(**{**train_cfg.__dict__, "objective": objective})
    hist = train_ssl(model, gen, cfg, pid_schedule=pid_schedule)
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
        "family3_kappa": float(cohen_kappa_score(yte_fam, fam_eval["pred"])),  # type: ignore[arg-type]
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


def _pid_confusion_failure_metrics(cm: np.ndarray) -> Dict[str, float]:
    """Compact pathology metrics for PID confusions: redundancy recall and R->S leakage."""
    cm = cm.astype(np.float64)
    row_sums = np.maximum(cm.sum(axis=1), 1.0)
    r_rows = np.array([3, 4, 5, 6], dtype=np.int64)  # R12, R13, R23, R123
    s_cols = np.array([7, 8, 9], dtype=np.int64)     # S12->3, S13->2, S23->1
    r_recalls = np.array([cm[i, i] / row_sums[i] for i in r_rows], dtype=np.float64)
    r_to_s = np.array([cm[i, s_cols].sum() / row_sums[i] for i in r_rows], dtype=np.float64)
    return {
        "r_recall_mean": float(np.mean(r_recalls)),
        "r_to_s_leakage_mean": float(np.mean(r_to_s)),
        "r_recall_r12": float(r_recalls[0]),
        "r_recall_r13": float(r_recalls[1]),
        "r_recall_r23": float(r_recalls[2]),
        "r_recall_r123": float(r_recalls[3]),
    }


def _mean_ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(max(n, 1))
    half = 1.96 * se if n > 1 else 0.0
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "se": float(se),
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
    }


def _slice_batch(batch: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray) and v.shape[0] == batch["pid_id"].shape[0]:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _fused_query_from_parts(parts: Dict[str, np.ndarray], keys: Tuple[str, ...]) -> np.ndarray:
    """
    Build same-dimensional query embeddings from one or more modality embeddings.
    For multi-source retrieval, average L2-normalized modality embeddings.
    """
    mats = [_l2_normalize_rows(parts[k].astype(np.float32)) for k in keys]
    if len(mats) == 1:
        return mats[0]
    return _l2_normalize_rows(np.mean(np.stack(mats, axis=0), axis=0))


def _retrieval_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = _l2_normalize_rows(query.astype(np.float32))
    g = _l2_normalize_rows(gallery.astype(np.float32))
    return (q @ g.T).astype(np.float32)


def _retrieval_metrics_from_scores(scores: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Positive for query i is gallery item i (same sample index).
    Returns per-query ranks and aggregate retrieval metrics.
    """
    n = int(scores.shape[0])
    # rank of the positive among descending scores
    pos = np.arange(n, dtype=np.int64)
    pos_scores = scores[pos, pos][:, None]
    better = np.sum(scores > pos_scores, axis=1).astype(np.int64)
    # break ties pessimistically by counting equal scores excluding self
    ties = np.sum(scores == pos_scores, axis=1).astype(np.int64) - 1
    rank = better + np.maximum(ties, 0) + 1  # 1-based rank
    r1 = float(np.mean(rank <= 1))
    r5 = float(np.mean(rank <= 5))
    mrr = float(np.mean(1.0 / rank.astype(np.float64)))
    return {
        "rank": rank,
        "recall_at_1": np.array([r1], dtype=np.float32),
        "recall_at_5": np.array([r5], dtype=np.float32),
        "mrr": np.array([mrr], dtype=np.float32),
    }


def _retrieval_metrics_stratified(rank: np.ndarray, pid_ids: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    fam_ids = family_from_pid_ids(pid_ids.astype(np.int64))
    for scope, labels, names in [
        ("family", np.arange(3, dtype=np.int64), FAMILY_NAMES),
        ("pid", np.arange(10, dtype=np.int64), PID_NAMES),
    ]:
        y = fam_ids if scope == "family" else pid_ids.astype(np.int64)
        for lab, name in zip(labels, names):
            m = y == int(lab)
            if not np.any(m):
                continue
            rk = rank[m].astype(np.float64)
            rows.append(
                {
                    "scope": 0.0,  # placeholders for typed dict
                    "label_id": float(lab),
                    "label_name": 0.0,
                    "n": float(rk.size),
                    "recall_at_1": float(np.mean(rk <= 1)),
                    "recall_at_5": float(np.mean(rk <= 5)),
                    "mrr": float(np.mean(1.0 / rk)),
                }
            )
            rows[-1]["scope"] = scope  # type: ignore[assignment]
            rows[-1]["label_name"] = name  # type: ignore[assignment]
    return rows


def _retrieval_metrics_from_rank_subset(rank: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    m = mask.astype(bool)
    if int(np.sum(m)) == 0:
        return {"n": 0.0, "recall_at_1": float("nan"), "recall_at_5": float("nan"), "mrr": float("nan")}
    rk = rank[m].astype(np.float64)
    return {
        "n": float(rk.size),
        "recall_at_1": float(np.mean(rk <= 1)),
        "recall_at_5": float(np.mean(rk <= 5)),
        "mrr": float(np.mean(1.0 / rk)),
    }


def _applicable_pid_ids_for_pair_to_target(source: str, target: str) -> np.ndarray:
    """
    PID atoms considered applicable for pair->heldout target prediction in the
    single-atom generator. A task is applicable if the target view receives
    signal that is a function of latents present in the source pair.
    """
    if len(source) != 2 or len(target) != 1 or target in source:
        raise ValueError(f"Applicability map is only defined for pair->heldout tasks, got {source}->{target}")
    sset = set(int(c) for c in source)
    t = int(target)
    out: List[int] = []
    # Pairwise redundancies Rij are applicable iff target participates in the pair.
    red_pairs = {3: (1, 2), 4: (1, 3), 5: (2, 3)}
    for pid_id, pair in red_pairs.items():
        pset = set(pair)
        if t in pset and len(pset & sset) >= 1:
            out.append(pid_id)
    # R123 is applicable for any pair->heldout task.
    out.append(6)
    # Directional synergy Sij->k is applicable exactly for its matching rotation.
    syn_map = {7: ("12", "3"), 8: ("13", "2"), 9: ("23", "1")}
    for pid_id, (src, tgt) in syn_map.items():
        if source == src and target == tgt:
            out.append(pid_id)
    return np.asarray(sorted(out), dtype=np.int64)


def _subset_concat(parts: Dict[str, np.ndarray], keys: Tuple[str, ...]) -> np.ndarray:
    return np.concatenate([parts[k] for k in keys], axis=1).astype(np.float32)


def _evaluate_subset_predictors(
    Xtr: np.ndarray,
    train_batch: Dict[str, np.ndarray],
    Xte: np.ndarray,
    test_batch: Dict[str, np.ndarray],
) -> List[Dict[str, float]]:
    parts_tr = _split_modalities_from_concat(Xtr)
    parts_te = _split_modalities_from_concat(Xte)
    subsets = [
        ("x1", ("x1",)),
        ("x2", ("x2",)),
        ("x3", ("x3",)),
        ("x12", ("x1", "x2")),
        ("x13", ("x1", "x3")),
        ("x23", ("x2", "x3")),
        ("x123", ("x1", "x2", "x3")),
    ]
    rows: List[Dict[str, float]] = []
    ytr_pid = train_batch["pid_id"].astype(np.int64)
    yte_pid = test_batch["pid_id"].astype(np.int64)
    ytr_fam = family_from_pid_ids(ytr_pid)
    yte_fam = family_from_pid_ids(yte_pid)
    for subset_name, ks in subsets:
        Xs_tr = _subset_concat(parts_tr, ks)
        Xs_te = _subset_concat(parts_te, ks)
        pid_eval = _fit_classifier_with_confusion(Xs_tr, ytr_pid, Xs_te, yte_pid, labels=np.arange(10))
        fam_eval = _fit_classifier_with_confusion(Xs_tr, ytr_fam, Xs_te, yte_fam, labels=np.arange(3))
        fam_pred = fam_eval["pred"]  # type: ignore[index]
        row: Dict[str, float] = {
            "pid10_acc": float(pid_eval["acc"][0]),
            "family3_acc": float(fam_eval["acc"][0]),
            "family3_kappa": float(cohen_kappa_score(yte_fam, fam_pred)),
            "family3_macro_f1": float(f1_score(yte_fam, fam_pred, average="macro", zero_division=0)),
            "u_f1": float(f1_score((yte_fam == 0).astype(np.int64), (fam_pred == 0).astype(np.int64), zero_division=0)),
            "r_f1": float(f1_score((yte_fam == 1).astype(np.int64), (fam_pred == 1).astype(np.int64), zero_division=0)),
            "s_f1": float(f1_score((yte_fam == 2).astype(np.int64), (fam_pred == 2).astype(np.int64), zero_division=0)),
        }
        for y_key, m_key in [("y_u1", "mask_y_u1"), ("y_r12", "mask_y_r12"), ("y_r123", "mask_y_r123"), ("y_s12_3", "mask_y_s12_3")]:
            mtr = train_batch[m_key].astype(bool)
            mte = test_batch[m_key].astype(bool)
            row[f"{y_key}_r2"] = _fit_ridge_r2(Xs_tr[mtr], train_batch[y_key][mtr], Xs_te[mte], test_batch[y_key][mte])
        row["subset"] = subset_name  # type: ignore[assignment]
        rows.append(row)
    return rows


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
    # Important: keep the same dataset seed across train/test probe generators so
    # fixed projections/synergy maps are shared; only sampled examples differ.
    probe_train_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
    probe_test_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
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
    - Model C: TRIANGLE (area contrastive)
    - Model D: ConFu (pairwise + trainable fused-pair-to-third contrastive terms)
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
    # Same fixed generator world for training and probe splits (shared projections/deleakage maps).
    probe_train_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
    probe_test_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
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

    model_c, hist_c = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "triangle_exact", "triangle_exact")
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
    subset_probe = {
        "model_a_unimodal_simclr_sum": _evaluate_subset_predictors(Xtr_a, probe_train, Xte_a, probe_test),
        "model_b_pairwise_infonce_sum": _evaluate_subset_predictors(Xtr_b, probe_train, Xte_b, probe_test),
        "model_c_triangle_like": _evaluate_subset_predictors(Xtr_c, probe_train, Xte_c, probe_test),
        "model_d_confu_style": _evaluate_subset_predictors(Xtr_d, probe_train, Xte_d, probe_test),
    }

    model_titles = {
        "model_a_unimodal_simclr_sum": "A: 3x unimodal SimCLR",
        "model_b_pairwise_infonce_sum": "B: pairwise InfoNCE sum\n(pairwise SimCLR/NT-Xent)",
        "model_c_triangle_like": "C: TRIANGLE",
        "model_d_confu_style": "D: ConFu",
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
    geo_labels = ["A: 3x uni SimCLR", "B: pairwise InfoNCE", "C: TRIANGLE", "D: ConFu"]
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

    # Subset predictor diagnostics (1/2/3 modality probes on frozen features)
    subset_order = ["x1", "x2", "x3", "x12", "x13", "x23", "x123"]
    metric_order = ["pid10_acc", "y_u1_r2", "y_r12_r2", "y_r123_r2", "y_s12_3_r2"]
    metric_labels = ["PID-10 acc", "R2(y_u1)", "R2(y_r12)", "R2(y_r123)", "R2(y_s12_3)"]
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 12.0))
    for ax, key in zip(axes.flat, geo_keys):
        row_map = {str(r["subset"]): r for r in subset_probe[key]}
        mat_sub = np.array([[float(row_map[s][m]) for m in metric_order] for s in subset_order], dtype=np.float32)
        im = ax.imshow(mat_sub, aspect="auto", cmap="coolwarm")
        ax.set_title(model_titles[key])
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=20, ha="right")
        ax.set_yticks(range(len(subset_order)))
        ax.set_yticklabels(subset_order)
        for i in range(mat_sub.shape[0]):
            for j in range(mat_sub.shape[1]):
                ax.text(j, i, f"{mat_sub[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.suptitle("Subset predictor diagnostics (1/2/3 modalities; frozen encoders + linear probes)", y=0.99)
    fig.subplots_adjust(bottom=0.15, wspace=0.28, hspace=0.30)
    _savefig(out_dir / "subset_predictor_heatmaps_four_models.png")

    # Family subset classification diagnostics (the most direct U/R/S grasp check).
    fam_metric_order = ["family3_acc", "family3_kappa", "family3_macro_f1", "u_f1", "r_f1", "s_f1"]
    fam_metric_labels = ["Family-3 acc", "Family-3 kappa", "Family-3 macro-F1", "U F1", "R F1", "S F1"]
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 12.0))
    for ax, key in zip(axes.flat, geo_keys):
        row_map = {str(r["subset"]): r for r in subset_probe[key]}
        mat_fam = np.array([[float(row_map[s][m]) for m in fam_metric_order] for s in subset_order], dtype=np.float32)
        im = ax.imshow(mat_fam, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_title(model_titles[key])
        ax.set_xticks(range(len(fam_metric_labels)))
        ax.set_xticklabels(fam_metric_labels, rotation=20, ha="right")
        ax.set_yticks(range(len(subset_order)))
        ax.set_yticklabels(subset_order)
        for i in range(mat_fam.shape[0]):
            for j in range(mat_fam.shape[1]):
                ax.text(j, i, f"{mat_fam[i, j]:.2f}", ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.suptitle("Subset family classification probes (U/R/S grasp check; frozen encoders + linear probes)", y=0.99)
    fig.subplots_adjust(bottom=0.15, wspace=0.28, hspace=0.30)
    _savefig(out_dir / "subset_family_probe_heatmaps_four_models.png")

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

    with (out_dir / "fused_frozen_four_models_subset_predictors.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "subset",
                "pid10_acc",
                "family3_acc",
                "family3_kappa",
                "family3_macro_f1",
                "u_f1",
                "r_f1",
                "s_f1",
                "y_u1_r2",
                "y_r12_r2",
                "y_r123_r2",
                "y_s12_3_r2",
            ]
        )
        for k in geo_keys:
            for r in subset_probe[k]:
                writer.writerow(
                    [
                        k,
                        r["subset"],
                        r["pid10_acc"],
                        r["family3_acc"],
                        r["family3_kappa"],
                        r["family3_macro_f1"],
                        r["u_f1"],
                        r["r_f1"],
                        r["s_f1"],
                        r["y_u1_r2"],
                        r["y_r12_r2"],
                        r["y_r123_r2"],
                        r["y_s12_3_r2"],
                    ]
                )

    with (out_dir / "fused_frozen_four_models_training_curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "stream", "step", "loss"])
        writer.writeheader()
        for r in hist_a + hist_b + hist_c + hist_d:
            writer.writerow({k: r.get(k, "") for k in ["model", "stream", "step", "loss"]})

    # Sanity checks
    for k in geo_keys:
        assert np.isfinite(float(evals[k]["pid10_acc"]))


def test_main_results_four_models_repeated_seed_summary():
    """
    Main-results quality upgrade:
    - repeated runs across generator worlds and optimization seeds
    - compact, decision-focused metrics
    - mean/std/95% CI instead of single-run leaderboard snapshots

    This is intentionally CSV-first. It avoids generating the full diagnostics stack
    and writes a compact uncertainty-aware summary for the main results document.
    """
    out_dir = _ensure_plot_dir()

    # Keep this modest for local CPU runs; increase repeats/steps for final reporting.
    repeats = 3
    data_seeds = [2201, 2202, 2203][:repeats]
    train_seeds = [71, 72, 73][:repeats]
    probe_train_shuffles = [101, 102, 103][:repeats]
    probe_test_shuffles = [201, 202, 203][:repeats]

    model_titles = {
        "model_a_unimodal_simclr_sum": "A: 3x unimodal SimCLR",
        "model_b_pairwise_infonce_sum": "B: pairwise InfoNCE sum",
        "model_c_triangle_like": "C: TRIANGLE",
        "model_d_confu_style": "D: ConFu",
    }
    primary_metrics = [
        "family3_acc",
        "family3_kappa",
        "r_recall_mean",
        "r_to_s_leakage_mean",
        "mean_matched_rs_centroid_cos",
    ]
    supplemental_metrics = [
        "mean_matched_rs_nc_acc",
        "y_r12_r2",
        "y_r123_r2",
        "y_s12_3_r2",
    ]

    raw_rows: List[Dict[str, float]] = []
    for trial_idx, (data_seed, train_seed, shuf_tr, shuf_te) in enumerate(
        zip(data_seeds, train_seeds, probe_train_shuffles, probe_test_shuffles), start=1
    ):
        data_cfg = PIDDatasetConfig(
            d=32,
            m=8,
            sigma=0.45,
            rho_choices=(0.2, 0.5, 0.8),
            hop_choices=(1, 2, 3, 4),
            seed=data_seed,
            deleakage_fit_samples=1024,
        )
        ssl_gen = PIDSar3DatasetGenerator(data_cfg)
        probe_train_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
        probe_test_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
        probe_train = _balanced_batch(probe_train_gen, n_per_pid=260, shuffle_seed=shuf_tr, return_aux=True)
        probe_test = _balanced_batch(probe_test_gen, n_per_pid=140, shuffle_seed=shuf_te, return_aux=True)

        enc_cfg = SSLEncoderConfig(
            input_dim=data_cfg.d,
            encoder_hidden_dim=96,
            representation_dim=48,
            projector_hidden_dim=96,
            projector_dim=48,
        )
        base_train_cfg = SSLTrainConfig(
            lr=1e-3,
            weight_decay=1e-5,
            batch_size=192,
            steps=140,
            temperature=0.2,
            device="cpu",
            seed=train_seed,
            triangle_reg_weight=0.15,
            confu_pair_weight=0.5,
            confu_fused_weight=0.5,
        )

        unimodal_models, _ = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, base_train_cfg)
        Xtr_a = _concat_unimodal_frozen(unimodal_models, probe_train)
        Xte_a = _concat_unimodal_frozen(unimodal_models, probe_test)

        model_b, _ = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "pairwise_simclr", "sum_3_pairwise_infonce")
        Xtr_b = _concat_trimodal_frozen(model_b, probe_train)
        Xte_b = _concat_trimodal_frozen(model_b, probe_test)

        model_c, _ = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "triangle_exact", "triangle_exact")
        Xtr_c = _concat_trimodal_frozen(model_c, probe_train)
        Xte_c = _concat_trimodal_frozen(model_c, probe_test)

        model_d, _ = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "confu_style", "confu_style")
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

        for model_key in model_titles:
            ev = evals[model_key]
            g = geoms[model_key]
            cm = ev["_pid_cm"]  # type: ignore[index]
            cm_metrics = _pid_confusion_failure_metrics(cm)
            row: Dict[str, float] = {
                "trial_idx": float(trial_idx),
                "data_seed": float(data_seed),
                "ssl_train_seed": float(train_seed),
                "probe_train_shuffle_seed": float(shuf_tr),
                "probe_test_shuffle_seed": float(shuf_te),
                "model": 0.0,  # placeholder for typed dict
                "pid10_acc": float(ev["pid10_acc"]),
                "family3_acc": float(ev["family3_acc"]),
                "family3_kappa": float(ev["family3_kappa"]),
                "y_r12_r2": float(ev["y_r12_r2"]),
                "y_r123_r2": float(ev["y_r123_r2"]),
                "y_s12_3_r2": float(ev["y_s12_3_r2"]),
                "mean_matched_rs_centroid_cos": float(np.mean(g["rs_pair_centroid_cos"])),
                "mean_matched_rs_nc_acc": float(np.mean(g["rs_pair_accs"])),
            }
            row.update(cm_metrics)
            row["model"] = model_key  # type: ignore[assignment]
            raw_rows.append(row)

    # Write per-trial rows (useful for appendix and debugging variance).
    raw_fieldnames = [
        "trial_idx",
        "data_seed",
        "ssl_train_seed",
        "probe_train_shuffle_seed",
        "probe_test_shuffle_seed",
        "model",
        "pid10_acc",
        *primary_metrics,
        *supplemental_metrics,
        "r_recall_r12",
        "r_recall_r13",
        "r_recall_r23",
        "r_recall_r123",
    ]
    with (out_dir / "main_results_four_models_seeded_trials.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)

    # Aggregate per model x metric.
    summary_rows: List[Dict[str, float]] = []
    for model_key in model_titles:
        model_rows = [r for r in raw_rows if str(r["model"]) == model_key]
        for metric in primary_metrics + supplemental_metrics:
            stats = _mean_ci95([float(r[metric]) for r in model_rows])
            summary_row: Dict[str, float] = {
                "model": 0.0,  # placeholder for typed dict
                "metric": 0.0,  # placeholder for typed dict
                "n_trials": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
                "se": stats["se"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
            }
            summary_row["model"] = model_key  # type: ignore[assignment]
            summary_row["metric"] = metric  # type: ignore[assignment]
            summary_rows.append(summary_row)

    with (out_dir / "main_results_four_models_seeded_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "metric", "n_trials", "mean", "std", "se", "ci95_low", "ci95_high"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # Main-results figure: compact error-bar panel for decision metrics only.
    fig, axes = plt.subplots(1, len(primary_metrics), figsize=(20.0, 4.8))
    x = np.arange(len(model_titles))
    model_keys = list(model_titles.keys())
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    pretty = {
        "family3_acc": "Family-3 acc",
        "family3_kappa": "Family-3 kappa",
        "r_recall_mean": "mean R recall",
        "r_to_s_leakage_mean": "mean R->S leakage",
        "mean_matched_rs_centroid_cos": "mean matched R/S centroid cos",
    }
    summary_map = {(str(r["model"]), str(r["metric"])): r for r in summary_rows}
    for ax, metric in zip(axes, primary_metrics):
        means = [float(summary_map[(k, metric)]["mean"]) for k in model_keys]
        errs = [float(summary_map[(k, metric)]["ci95_high"]) - float(summary_map[(k, metric)]["mean"]) for k in model_keys]
        ax.bar(x, means, color=colors, alpha=0.9)
        ax.errorbar(x, means, yerr=errs, fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["A", "B", "C", "D"])
        ax.set_title(pretty[metric], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if metric in ("r_to_s_leakage_mean", "mean_matched_rs_centroid_cos"):
            ax.set_ylabel("lower is better")
        else:
            ax.set_ylabel("higher is better")
    fig.suptitle("Main results (4 models): repeated-seed summary with 95% CIs", y=1.02)
    fig.tight_layout()
    _savefig(out_dir / "main_results_four_models_seeded_summary.png")

    assert len(raw_rows) == repeats * 4


def test_source_to_target_four_models_kappa_5fold():
    """
    Regenerate source->target results with 5-fold probe evaluation and export macro-kappa.

    This produces a table-ready artifact for the main results section (`7a/7/7b`)
    with kappa as the primary metric and macro-F1 retained for comparability.
    """
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.45,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=3401,
        deleakage_fit_samples=1024,
    )
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)
    enc_cfg = SSLEncoderConfig(input_dim=data_cfg.d, encoder_hidden_dim=96, representation_dim=48, projector_hidden_dim=96, projector_dim=48)
    base_train_cfg = SSLTrainConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=192,
        steps=120,
        temperature=0.2,
        device="cpu",
        seed=91,
        triangle_reg_weight=0.15,
        confu_pair_weight=0.5,
        confu_fused_weight=0.5,
    )

    # Train once, evaluate on 5 probe folds (same world).
    unimodal_models, _ = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, base_train_cfg)
    model_b, _ = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "pairwise_simclr", "sum_3_pairwise_infonce")
    model_c, _ = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "triangle_exact", "triangle_exact")
    model_d, _ = _train_trimodal_objective(ssl_gen, enc_cfg, base_train_cfg, "confu_style", "confu_style")

    probe_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
    probe_all = _balanced_batch(probe_gen, n_per_pid=250, shuffle_seed=777, return_aux=True)
    y_all_pid = probe_all["pid_id"].astype(np.int64)

    X_all = {
        "A": _concat_unimodal_frozen(unimodal_models, probe_all),
        "B": _concat_trimodal_frozen(model_b, probe_all),
        "C": _concat_trimodal_frozen(model_c, probe_all),
        "D": _concat_trimodal_frozen(model_d, probe_all),
    }
    model_labels = {
        "A": "A: 3x unimodal SimCLR",
        "B": "B: pairwise InfoNCE",
        "C": "C: TRIANGLE",
        "D": "D: ConFu",
    }
    source_map = {
        "1": ("x1",),
        "2": ("x2",),
        "3": ("x3",),
        "12": ("x1", "x2"),
        "13": ("x1", "x3"),
        "23": ("x2", "x3"),
        "123": ("x1", "x2", "x3"),
    }
    target_keys = {"1": "x1", "2": "x2", "3": "x3"}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    fold_rows: List[Dict[str, float]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros_like(y_all_pid), y_all_pid), start=1):
        batch_tr = _slice_batch(probe_all, tr_idx)
        batch_te = _slice_batch(probe_all, te_idx)
        for mkey, X in X_all.items():
            Xtr = X[tr_idx]
            Xte = X[te_idx]
            parts_tr = _split_modalities_from_concat(Xtr)
            parts_te = _split_modalities_from_concat(Xte)
            for src, src_keys in source_map.items():
                Xsrc_tr = _subset_concat(parts_tr, src_keys)
                Xsrc_te = _subset_concat(parts_te, src_keys)
                for tgt in ("1", "2", "3"):
                    metrics = _fit_binary_macro_f1_kappa_over_target_dims(
                        Xsrc_tr,
                        batch_tr[target_keys[tgt]].astype(np.float32),
                        Xsrc_te,
                        batch_te[target_keys[tgt]].astype(np.float32),
                    )
                    fold_rows.append(
                        {
                            "fold": float(fold_idx),
                            "model": 0.0,  # placeholder
                            "model_label": 0.0,  # placeholder
                            "source": 0.0,  # placeholder
                            "target": 0.0,  # placeholder
                            "macro_f1": float(metrics["macro_f1"]),
                            "macro_kappa": float(metrics["macro_kappa"]),
                            "n_target_dims": float(metrics["n_target_dims"]),
                        }
                    )
                    fold_rows[-1]["model"] = mkey  # type: ignore[assignment]
                    fold_rows[-1]["model_label"] = model_labels[mkey]  # type: ignore[assignment]
                    fold_rows[-1]["source"] = src  # type: ignore[assignment]
                    fold_rows[-1]["target"] = tgt  # type: ignore[assignment]

    with (out_dir / "source_to_target_four_models_5fold_per_fold.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "model", "model_label", "source", "target", "macro_f1", "macro_kappa", "n_target_dims"],
        )
        writer.writeheader()
        for r in fold_rows:
            writer.writerow(r)

    # Aggregate mean ± SE by (model, source, target)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}
    for r in fold_rows:
        key = (str(r["model"]), str(r["source"]), str(r["target"]))
        grouped.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, float]] = []
    for (m, src, tgt), rows in grouped.items():
        f1_stats = _mean_ci95([float(r["macro_f1"]) for r in rows])
        k_stats = _mean_ci95([float(r["macro_kappa"]) for r in rows])
        summary_rows.append(
            {
                "model": 0.0,  # placeholder
                "model_label": 0.0,  # placeholder
                "source": 0.0,  # placeholder
                "target": 0.0,  # placeholder
                "n_folds": float(len(rows)),
                "macro_f1_mean": float(f1_stats["mean"]),
                "macro_f1_se": float(f1_stats["se"]),
                "macro_kappa_mean": float(k_stats["mean"]),
                "macro_kappa_se": float(k_stats["se"]),
            }
        )
        summary_rows[-1]["model"] = m  # type: ignore[assignment]
        summary_rows[-1]["model_label"] = model_labels[m]  # type: ignore[assignment]
        summary_rows[-1]["source"] = src  # type: ignore[assignment]
        summary_rows[-1]["target"] = tgt  # type: ignore[assignment]

    summary_rows.sort(key=lambda r: (str(r["model"]), len(str(r["source"])), str(r["source"]), str(r["target"])))
    with (out_dir / "source_to_target_four_models_5fold_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "model_label",
                "source",
                "target",
                "n_folds",
                "macro_f1_mean",
                "macro_f1_se",
                "macro_kappa_mean",
                "macro_kappa_se",
            ],
        )
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # Grouped 7a-style summary on kappa and F1
    def _group_name(src: str, tgt: str) -> str:
        if len(src) == 1 and src == tgt:
            return "self_1to1"
        if len(src) == 1 and src != tgt:
            return "single_cross"
        if len(src) == 2 and tgt not in src:
            return "pair_to_heldout_target"
        if len(src) == 2 and tgt in src:
            return "pair_to_member_target"
        if len(src) == 3:
            return "triple_to_target"
        return "other"

    grouped_rows: List[Dict[str, float]] = []
    by_model_group: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for r in summary_rows:
        g = _group_name(str(r["source"]), str(r["target"]))
        by_model_group.setdefault((str(r["model"]), g), []).append(r)
    for (m, g), rows in by_model_group.items():
        grouped_rows.append(
            {
                "model": 0.0,
                "model_label": 0.0,
                "group": 0.0,
                "macro_f1_mean": float(np.mean([float(r["macro_f1_mean"]) for r in rows])),
                "macro_kappa_mean": float(np.mean([float(r["macro_kappa_mean"]) for r in rows])),
            }
        )
        grouped_rows[-1]["model"] = m  # type: ignore[assignment]
        grouped_rows[-1]["model_label"] = model_labels[m]  # type: ignore[assignment]
        grouped_rows[-1]["group"] = g  # type: ignore[assignment]
    with (out_dir / "source_to_target_four_models_5fold_grouped_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "model_label", "group", "macro_f1_mean", "macro_kappa_mean"])
        writer.writeheader()
        for r in grouped_rows:
            writer.writerow(r)

    assert len(summary_rows) == 4 * 7 * 3


def test_retrieval_source_to_target_four_models():
    """
    PID-stratified cross-modal retrieval benchmark on frozen embeddings.

    Query embeddings are built from source subsets by averaging normalized modality
    embeddings (same embedding dimensionality), and retrieval is performed against
    the target modality gallery using cosine similarity.
    """
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.45,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=3501,
        deleakage_fit_samples=1024,
    )
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)
    enc_cfg = SSLEncoderConfig(input_dim=data_cfg.d, encoder_hidden_dim=96, representation_dim=48, projector_hidden_dim=96, projector_dim=48)
    train_cfg = SSLTrainConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=192,
        steps=180,
        temperature=0.2,
        device="cpu",
        seed=101,
        triangle_reg_weight=0.15,
        confu_pair_weight=0.5,
        confu_fused_weight=0.5,
    )

    # Train once (same-world SSL), evaluate retrieval on held-out probe samples.
    unimodal_models, _ = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, train_cfg)
    model_b, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "pairwise_simclr", "sum_3_pairwise_infonce")
    model_c, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "triangle_exact", "triangle_exact")
    model_d, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "confu_style", "confu_style")

    probe_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
    probe = _balanced_batch(probe_gen, n_per_pid=180, shuffle_seed=909, return_aux=True)
    pid_ids = probe["pid_id"].astype(np.int64)

    Xs = {
        "A": _concat_unimodal_frozen(unimodal_models, probe),
        "B": _concat_trimodal_frozen(model_b, probe),
        "C": _concat_trimodal_frozen(model_c, probe),
        "D": _concat_trimodal_frozen(model_d, probe),
    }
    model_labels = {
        "A": "A: 3x unimodal SimCLR",
        "B": "B: pairwise InfoNCE",
        "C": "C: TRIANGLE",
        "D": "D: ConFu",
    }
    source_map = {
        "1": ("x1",),
        "2": ("x2",),
        "3": ("x3",),
        "12": ("x1", "x2"),
        "13": ("x1", "x3"),
        "23": ("x2", "x3"),
        "123": ("x1", "x2", "x3"),
    }
    target_keys = {"1": "x1", "2": "x2", "3": "x3"}

    pair_rows: List[Dict[str, float]] = []
    strat_rows: List[Dict[str, float]] = []
    for mkey, X in Xs.items():
        parts = _split_modalities_from_concat(X)
        for src, src_keys in source_map.items():
            query = _fused_query_from_parts(parts, src_keys)
            for tgt, tgt_key in target_keys.items():
                gallery = parts[tgt_key]
                scores = _retrieval_scores(query, gallery)
                metrics = _retrieval_metrics_from_scores(scores)
                rank = metrics["rank"]  # type: ignore[index]
                pair_rows.append(
                    {
                        "model": 0.0,
                        "model_label": 0.0,
                        "source": 0.0,
                        "target": 0.0,
                        "recall_at_1": float(metrics["recall_at_1"][0]),
                        "recall_at_5": float(metrics["recall_at_5"][0]),
                        "mrr": float(metrics["mrr"][0]),
                        "n": float(rank.shape[0]),
                    }
                )
                pair_rows[-1]["model"] = mkey  # type: ignore[assignment]
                pair_rows[-1]["model_label"] = model_labels[mkey]  # type: ignore[assignment]
                pair_rows[-1]["source"] = src  # type: ignore[assignment]
                pair_rows[-1]["target"] = tgt  # type: ignore[assignment]

                for r in _retrieval_metrics_stratified(rank.astype(np.int64), pid_ids):
                    rr = dict(r)
                    rr.update({"model": mkey, "model_label": model_labels[mkey], "source": src, "target": tgt})
                    strat_rows.append(rr)

    with (out_dir / "retrieval_source_to_target_four_models_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "model_label", "source", "target", "recall_at_1", "recall_at_5", "mrr", "n"],
        )
        writer.writeheader()
        for r in pair_rows:
            writer.writerow(r)

    with (out_dir / "retrieval_source_to_target_four_models_stratified.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "model_label",
                "source",
                "target",
                "scope",
                "label_id",
                "label_name",
                "n",
                "recall_at_1",
                "recall_at_5",
                "mrr",
            ],
        )
        writer.writeheader()
        for r in strat_rows:
            writer.writerow(r)

    # Compact heatmap for main source->target retrieval summary (Recall@1 by default).
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.0))
    src_order = ["12", "13", "23", "123"]
    tgt_order = ["1", "2", "3"]
    for ax, mkey in zip(axes.flat, ["A", "B", "C", "D"]):
        row_map = {(str(r["source"]), str(r["target"])): r for r in pair_rows if str(r["model"]) == mkey}
        mat = np.array([[float(row_map[(s, t)]["recall_at_1"]) for t in tgt_order] for s in src_order], dtype=np.float32)
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_title(model_labels[mkey])
        ax.set_xticks(range(len(tgt_order)))
        ax.set_xticklabels([f"target {t}" for t in tgt_order])
        ax.set_yticks(range(len(src_order)))
        ax.set_yticklabels([f"src {s}" for s in src_order])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.suptitle("Frozen-embedding source->target retrieval (Recall@1)", y=0.99)
    fig.subplots_adjust(wspace=0.25, hspace=0.28)
    _savefig(out_dir / "retrieval_source_to_target_four_models_recall1_heatmaps.png")

    assert len(pair_rows) == 4 * 7 * 3


def test_source_to_target_reconstruction_four_models_5fold():
    """
    Frozen-encoder source->target reconstruction with multi-output decoders.

    We compare a linear decoder (Ridge) and a nonlinear decoder (MLP) on the same
    5-fold source->target grid used for the kappa benchmark, reporting macro R^2
    and macro normalized RMSE over target dimensions.
    """
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.45,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=3601,
        deleakage_fit_samples=1024,
    )
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)
    enc_cfg = SSLEncoderConfig(input_dim=data_cfg.d, encoder_hidden_dim=96, representation_dim=48, projector_hidden_dim=96, projector_dim=48)
    train_cfg = SSLTrainConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=192,
        steps=180,
        temperature=0.2,
        device="cpu",
        seed=111,
        triangle_reg_weight=0.15,
        confu_pair_weight=0.5,
        confu_fused_weight=0.5,
    )

    unimodal_models, _ = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, train_cfg)
    model_b, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "pairwise_simclr", "sum_3_pairwise_infonce")
    model_c, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "triangle_exact", "triangle_exact")
    model_d, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "confu_style", "confu_style")

    probe_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
    probe_all = _balanced_batch(probe_gen, n_per_pid=60, shuffle_seed=919, return_aux=True)
    y_all_pid = probe_all["pid_id"].astype(np.int64)

    X_all = {
        "A": _concat_unimodal_frozen(unimodal_models, probe_all),
        "B": _concat_trimodal_frozen(model_b, probe_all),
        "C": _concat_trimodal_frozen(model_c, probe_all),
        "D": _concat_trimodal_frozen(model_d, probe_all),
    }
    model_labels = {
        "A": "A: 3x unimodal SimCLR",
        "B": "B: pairwise InfoNCE",
        "C": "C: TRIANGLE",
        "D": "D: ConFu",
    }
    source_map = {
        "12": ("x1", "x2"),
        "13": ("x1", "x3"),
        "23": ("x2", "x3"),
        "123": ("x1", "x2", "x3"),
    }
    target_keys = {"1": "x1", "2": "x2", "3": "x3"}
    decoder_labels = {"ridge": "Ridge", "mlp": "MLP"}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=19)
    fold_rows: List[Dict[str, float]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros_like(y_all_pid), y_all_pid), start=1):
        batch_tr = _slice_batch(probe_all, tr_idx)
        batch_te = _slice_batch(probe_all, te_idx)
        for mkey, X in X_all.items():
            Xtr = X[tr_idx]
            Xte = X[te_idx]
            parts_tr = _split_modalities_from_concat(Xtr)
            parts_te = _split_modalities_from_concat(Xte)
            for src, src_keys in source_map.items():
                Xsrc_tr = _subset_concat(parts_tr, src_keys)
                Xsrc_te = _subset_concat(parts_te, src_keys)
                for tgt in ("1", "2", "3"):
                    Ytr = batch_tr[target_keys[tgt]].astype(np.float32)
                    Yte = batch_te[target_keys[tgt]].astype(np.float32)
                    for dec_key in ("ridge", "mlp"):
                        metrics = _fit_reconstruction_decoder_metrics(Xsrc_tr, Ytr, Xsrc_te, Yte, decoder=dec_key)
                        fold_rows.append(
                            {
                                "fold": float(fold_idx),
                                "model": 0.0,
                                "model_label": 0.0,
                                "decoder": 0.0,
                                "decoder_label": 0.0,
                                "source": 0.0,
                                "target": 0.0,
                                "macro_r2": float(metrics["macro_r2"]),
                                "macro_nrmse": float(metrics["macro_nrmse"]),
                                "n_target_dims": float(metrics["n_target_dims"]),
                            }
                        )
                        fold_rows[-1]["model"] = mkey  # type: ignore[assignment]
                        fold_rows[-1]["model_label"] = model_labels[mkey]  # type: ignore[assignment]
                        fold_rows[-1]["decoder"] = dec_key  # type: ignore[assignment]
                        fold_rows[-1]["decoder_label"] = decoder_labels[dec_key]  # type: ignore[assignment]
                        fold_rows[-1]["source"] = src  # type: ignore[assignment]
                        fold_rows[-1]["target"] = tgt  # type: ignore[assignment]

    with (out_dir / "source_to_target_reconstruction_four_models_5fold_per_fold.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "model",
                "model_label",
                "decoder",
                "decoder_label",
                "source",
                "target",
                "macro_r2",
                "macro_nrmse",
                "n_target_dims",
            ],
        )
        writer.writeheader()
        for r in fold_rows:
            writer.writerow(r)

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, float]]] = {}
    for r in fold_rows:
        key = (str(r["model"]), str(r["decoder"]), str(r["source"]), str(r["target"]))
        grouped.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, float]] = []
    for (m, dec, src, tgt), rows in grouped.items():
        r2_stats = _mean_ci95([float(r["macro_r2"]) for r in rows])
        nr_stats = _mean_ci95([float(r["macro_nrmse"]) for r in rows])
        summary_rows.append(
            {
                "model": 0.0,
                "model_label": 0.0,
                "decoder": 0.0,
                "decoder_label": 0.0,
                "source": 0.0,
                "target": 0.0,
                "n_folds": float(len(rows)),
                "macro_r2_mean": float(r2_stats["mean"]),
                "macro_r2_se": float(r2_stats["se"]),
                "macro_nrmse_mean": float(nr_stats["mean"]),
                "macro_nrmse_se": float(nr_stats["se"]),
            }
        )
        summary_rows[-1]["model"] = m  # type: ignore[assignment]
        summary_rows[-1]["model_label"] = model_labels[m]  # type: ignore[assignment]
        summary_rows[-1]["decoder"] = dec  # type: ignore[assignment]
        summary_rows[-1]["decoder_label"] = decoder_labels[dec]  # type: ignore[assignment]
        summary_rows[-1]["source"] = src  # type: ignore[assignment]
        summary_rows[-1]["target"] = tgt  # type: ignore[assignment]

    summary_rows.sort(key=lambda r: (str(r["decoder"]), str(r["model"]), len(str(r["source"])), str(r["source"]), str(r["target"])))
    with (out_dir / "source_to_target_reconstruction_four_models_5fold_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "model_label",
                "decoder",
                "decoder_label",
                "source",
                "target",
                "n_folds",
                "macro_r2_mean",
                "macro_r2_se",
                "macro_nrmse_mean",
                "macro_nrmse_se",
            ],
        )
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    def _group_name(src: str, tgt: str) -> str:
        if len(src) == 1 and src == tgt:
            return "self_1to1"
        if len(src) == 1 and src != tgt:
            return "single_cross"
        if len(src) == 2 and tgt not in src:
            return "pair_to_heldout_target"
        if len(src) == 2 and tgt in src:
            return "pair_to_member_target"
        if len(src) == 3:
            return "triple_to_target"
        return "other"

    grouped_rows: List[Dict[str, float]] = []
    by_model_group: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}
    for r in summary_rows:
        g = _group_name(str(r["source"]), str(r["target"]))
        by_model_group.setdefault((str(r["decoder"]), str(r["model"]), g), []).append(r)
    for (dec, m, g), rows in by_model_group.items():
        grouped_rows.append(
            {
                "decoder": 0.0,
                "decoder_label": 0.0,
                "model": 0.0,
                "model_label": 0.0,
                "group": 0.0,
                "macro_r2_mean": float(np.mean([float(r["macro_r2_mean"]) for r in rows])),
                "macro_nrmse_mean": float(np.mean([float(r["macro_nrmse_mean"]) for r in rows])),
            }
        )
        grouped_rows[-1]["decoder"] = dec  # type: ignore[assignment]
        grouped_rows[-1]["decoder_label"] = decoder_labels[dec]  # type: ignore[assignment]
        grouped_rows[-1]["model"] = m  # type: ignore[assignment]
        grouped_rows[-1]["model_label"] = model_labels[m]  # type: ignore[assignment]
        grouped_rows[-1]["group"] = g  # type: ignore[assignment]

    with (out_dir / "source_to_target_reconstruction_four_models_5fold_grouped_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["decoder", "decoder_label", "model", "model_label", "group", "macro_r2_mean", "macro_nrmse_mean"],
        )
        writer.writeheader()
        for r in grouped_rows:
            writer.writerow(r)

    fig, axes = plt.subplots(2, 4, figsize=(20.0, 10.5), sharex=True, sharey=True)
    src_order = ["12", "13", "23", "123"]
    tgt_order = ["1", "2", "3"]
    for row_i, dec_key in enumerate(["ridge", "mlp"]):
        for col_i, mkey in enumerate(["A", "B", "C", "D"]):
            ax = axes[row_i, col_i]
            row_map = {
                (str(r["source"]), str(r["target"])): r
                for r in summary_rows
                if str(r["decoder"]) == dec_key and str(r["model"]) == mkey
            }
            mat = np.array([[float(row_map[(s, t)]["macro_r2_mean"]) for t in tgt_order] for s in src_order], dtype=np.float32)
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
            title = f"{decoder_labels[dec_key]} | {model_labels[mkey]}"
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(len(tgt_order)))
            ax.set_xticklabels([f"t{t}" for t in tgt_order], fontsize=8)
            ax.set_yticks(range(len(src_order)))
            ax.set_yticklabels([f"s{s}" for s in src_order], fontsize=8)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.suptitle("Frozen-encoder source->target reconstruction (macro R^2, 5-fold mean)", y=0.995)
    fig.subplots_adjust(wspace=0.18, hspace=0.28)
    _savefig(out_dir / "source_to_target_reconstruction_four_models_5fold_macro_r2_heatmaps.png")

    assert len(summary_rows) == 2 * 4 * 4 * 3


def test_pair_to_heldout_retrieval_applicability_low_noise():
    """
    Low-noise pathology diagnostic for pair->heldout retrieval.

    We report exact-instance retrieval on the rotated pair->heldout tasks under
    very low observation noise, split into applicable vs non-applicable PID atoms
    in the single-atom generator.
    """
    out_dir = _ensure_plot_dir()

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.05,  # very low noise to test whether the pathology is structural
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=3701,
        deleakage_fit_samples=1024,
    )
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)
    enc_cfg = SSLEncoderConfig(input_dim=data_cfg.d, encoder_hidden_dim=96, representation_dim=48, projector_hidden_dim=96, projector_dim=48)
    train_cfg = SSLTrainConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=192,
        steps=180,
        temperature=0.2,
        device="cpu",
        seed=121,
        triangle_reg_weight=0.15,
        confu_pair_weight=0.5,
        confu_fused_weight=0.5,
    )

    unimodal_models, _ = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, train_cfg)
    model_b, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "pairwise_simclr", "sum_3_pairwise_infonce")
    model_c, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "triangle_exact", "triangle_exact")
    model_d, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "confu_style", "confu_style")

    probe_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
    probe = _balanced_batch(probe_gen, n_per_pid=180, shuffle_seed=929, return_aux=True)
    pid_ids = probe["pid_id"].astype(np.int64)

    Xs = {
        "RAW": _concat_raw(probe),
        "A": _concat_unimodal_frozen(unimodal_models, probe),
        "B": _concat_trimodal_frozen(model_b, probe),
        "C": _concat_trimodal_frozen(model_c, probe),
        "D": _concat_trimodal_frozen(model_d, probe),
    }
    model_labels = {
        "RAW": "RAW: observations",
        "A": "A: 3x unimodal SimCLR",
        "B": "B: pairwise InfoNCE",
        "C": "C: TRIANGLE",
        "D": "D: ConFu",
    }
    rotations = [("23", "1"), ("13", "2"), ("12", "3")]

    rows: List[Dict[str, float]] = []
    for mkey, X in Xs.items():
        parts = _split_modalities_from_concat(X)
        for src, tgt in rotations:
            query = _fused_query_from_parts(parts, tuple(f"x{k}" for k in src))
            gallery = parts[f"x{tgt}"]
            scores = _retrieval_scores(query, gallery)
            metrics = _retrieval_metrics_from_scores(scores)
            rank = metrics["rank"].astype(np.int64)  # type: ignore[index]

            applicable_ids = _applicable_pid_ids_for_pair_to_target(src, tgt)
            m_app = np.isin(pid_ids, applicable_ids)
            m_non = ~m_app
            split_metrics = {
                "all": _retrieval_metrics_from_rank_subset(rank, np.ones_like(pid_ids, dtype=bool)),
                "applicable": _retrieval_metrics_from_rank_subset(rank, m_app),
                "non_applicable": _retrieval_metrics_from_rank_subset(rank, m_non),
            }

            for split_name, sm in split_metrics.items():
                rows.append(
                    {
                        "model": 0.0,
                        "model_label": 0.0,
                        "source": 0.0,
                        "target": 0.0,
                        "split": 0.0,
                        "n": float(sm["n"]),
                        "recall_at_1": float(sm["recall_at_1"]),
                        "recall_at_5": float(sm["recall_at_5"]),
                        "mrr": float(sm["mrr"]),
                        "random_recall_at_1": float(1.0 / rank.shape[0]),
                        "applicable_rate": float(np.mean(m_app)),
                    }
                )
                rows[-1]["model"] = mkey  # type: ignore[assignment]
                rows[-1]["model_label"] = model_labels[mkey]  # type: ignore[assignment]
                rows[-1]["source"] = src  # type: ignore[assignment]
                rows[-1]["target"] = tgt  # type: ignore[assignment]
                rows[-1]["split"] = split_name  # type: ignore[assignment]

    with (out_dir / "pair_to_heldout_retrieval_applicability_low_noise.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "model_label",
                "source",
                "target",
                "split",
                "n",
                "recall_at_1",
                "recall_at_5",
                "mrr",
                "random_recall_at_1",
                "applicable_rate",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Compact figure: mean Recall@1 across rotations for all/applicable/non-applicable.
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    split_order = ["all", "applicable", "non_applicable"]
    colors = ["#7f7f7f", "#2ca02c", "#d62728"]
    for ax, split_name, color in zip(axes, split_order, colors):
        x = [r for r in rows if str(r["split"]) == split_name]
        means = []
        labels = ["RAW", "A", "B", "C", "D"]
        for m in labels:
            vals = [float(r["recall_at_1"]) for r in x if str(r["model"]) == m]
            means.append(float(np.mean(vals)))
        ax.bar(np.arange(len(labels)), means, color=color, alpha=0.85)
        ax.axhline(float(1.0 / int(rows[0]["n"])), color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(split_name.replace("_", " "))
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, max(0.05, max(means) * 1.15))
        ax.grid(axis="y", alpha=0.25)
        if split_name == "all":
            ax.set_ylabel("Recall@1")
    fig.suptitle("Low-noise pair->heldout retrieval pathology (exact-instance, mean over rotations)", y=1.02)
    fig.tight_layout()
    _savefig(out_dir / "pair_to_heldout_retrieval_applicability_low_noise.png")

    assert len(rows) == 5 * 3 * 3


def test_pair_to_heldout_retrieval_applicability_low_noise_redundancy_train_only():
    """
    Same low-noise pathology diagnostic, but SSL training batches are restricted to
    redundancy atoms only (R12, R13, R23, R123).

    This tests whether the failure is mainly caused by mixed-atom supervision noise.
    """
    out_dir = _ensure_plot_dir()
    redundancy_only = (3, 4, 5, 6)

    data_cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.05,
        rho_choices=(0.2, 0.5, 0.8),
        hop_choices=(1, 2, 3, 4),
        seed=3711,
        deleakage_fit_samples=1024,
    )
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)
    enc_cfg = SSLEncoderConfig(input_dim=data_cfg.d, encoder_hidden_dim=96, representation_dim=48, projector_hidden_dim=96, projector_dim=48)
    train_cfg = SSLTrainConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=192,
        steps=180,
        temperature=0.2,
        device="cpu",
        seed=131,
        triangle_reg_weight=0.15,
        confu_pair_weight=0.5,
        confu_fused_weight=0.5,
    )

    unimodal_models, _ = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, train_cfg, pid_schedule=redundancy_only)
    model_b, _ = _train_trimodal_objective(
        ssl_gen, enc_cfg, train_cfg, "pairwise_simclr", "sum_3_pairwise_infonce", pid_schedule=redundancy_only
    )
    model_c, _ = _train_trimodal_objective(
        ssl_gen, enc_cfg, train_cfg, "triangle_exact", "triangle_exact", pid_schedule=redundancy_only
    )
    model_d, _ = _train_trimodal_objective(
        ssl_gen, enc_cfg, train_cfg, "confu_style", "confu_style", pid_schedule=redundancy_only
    )

    probe_gen = PIDSar3DatasetGenerator(PIDDatasetConfig(**{**data_cfg.__dict__, "seed": data_cfg.seed}))
    probe = _balanced_batch(probe_gen, n_per_pid=180, shuffle_seed=939, return_aux=True)
    pid_ids = probe["pid_id"].astype(np.int64)

    Xs = {
        "RAW": _concat_raw(probe),
        "A": _concat_unimodal_frozen(unimodal_models, probe),
        "B": _concat_trimodal_frozen(model_b, probe),
        "C": _concat_trimodal_frozen(model_c, probe),
        "D": _concat_trimodal_frozen(model_d, probe),
    }
    model_labels = {
        "RAW": "RAW: observations",
        "A": "A: 3x unimodal SimCLR (R-only train)",
        "B": "B: pairwise InfoNCE (R-only train)",
        "C": "C: TRIANGLE (R-only train)",
        "D": "D: ConFu (R-only train)",
    }
    rotations = [("23", "1"), ("13", "2"), ("12", "3")]

    rows: List[Dict[str, float]] = []
    for mkey, X in Xs.items():
        parts = _split_modalities_from_concat(X)
        for src, tgt in rotations:
            query = _fused_query_from_parts(parts, tuple(f"x{k}" for k in src))
            gallery = parts[f"x{tgt}"]
            scores = _retrieval_scores(query, gallery)
            rank = _retrieval_metrics_from_scores(scores)["rank"].astype(np.int64)  # type: ignore[index]

            applicable_ids = _applicable_pid_ids_for_pair_to_target(src, tgt)
            m_app = np.isin(pid_ids, applicable_ids)
            m_non = ~m_app
            split_metrics = {
                "all": _retrieval_metrics_from_rank_subset(rank, np.ones_like(pid_ids, dtype=bool)),
                "applicable": _retrieval_metrics_from_rank_subset(rank, m_app),
                "non_applicable": _retrieval_metrics_from_rank_subset(rank, m_non),
            }
            for split_name, sm in split_metrics.items():
                rows.append(
                    {
                        "model": 0.0,
                        "model_label": 0.0,
                        "source": 0.0,
                        "target": 0.0,
                        "split": 0.0,
                        "n": float(sm["n"]),
                        "recall_at_1": float(sm["recall_at_1"]),
                        "recall_at_5": float(sm["recall_at_5"]),
                        "mrr": float(sm["mrr"]),
                        "random_recall_at_1": float(1.0 / rank.shape[0]),
                        "applicable_rate": float(np.mean(m_app)),
                        "train_pid_subset": 0.0,
                    }
                )
                rows[-1]["model"] = mkey  # type: ignore[assignment]
                rows[-1]["model_label"] = model_labels[mkey]  # type: ignore[assignment]
                rows[-1]["source"] = src  # type: ignore[assignment]
                rows[-1]["target"] = tgt  # type: ignore[assignment]
                rows[-1]["split"] = split_name  # type: ignore[assignment]
                rows[-1]["train_pid_subset"] = "redundancy_only"  # type: ignore[assignment]

    with (out_dir / "pair_to_heldout_retrieval_applicability_low_noise_redundancy_train_only.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "model_label",
                "source",
                "target",
                "split",
                "n",
                "recall_at_1",
                "recall_at_5",
                "mrr",
                "random_recall_at_1",
                "applicable_rate",
                "train_pid_subset",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    assert len(rows) == 5 * 3 * 3


def test_compositional_easy_raw_retrieval_sanity():
    """
    Sanity-check a compositional dataset mode where cross-modal retrieval should be
    feasible before increasing difficulty.

    This is a dataset benchmarkability test (RAW observations only), not an SSL
    model comparison.
    """
    out_dir = _ensure_plot_dir()
    cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=0.03,
        rho_choices=(0.5, 0.8),
        hop_choices=(1, 2),
        seed=3801,
        deleakage_fit_samples=1024,
        composition_mode="multi_atom",
        active_atoms_per_sample=4,
        shared_backbone_gain=2.5,
        shared_backbone_tied_projection=True,
        synergy_deleak_lambda=0.5,
    )
    gen = PIDSar3DatasetGenerator(cfg)
    batch = _balanced_batch(gen, n_per_pid=180, shuffle_seed=949, return_aux=True)
    pid_ids = batch["pid_id"].astype(np.int64)  # primary atom; used only for applicability split
    X = _concat_raw(batch)
    parts = _split_modalities_from_concat(X)

    source_map = {"12": ("x1", "x2"), "13": ("x1", "x3"), "23": ("x2", "x3"), "123": ("x1", "x2", "x3")}
    target_keys = {"1": "x1", "2": "x2", "3": "x3"}
    rows: List[Dict[str, float]] = []
    for src, src_keys in source_map.items():
        query = _fused_query_from_parts(parts, src_keys)
        for tgt, tgt_key in target_keys.items():
            gallery = parts[tgt_key]
            scores = _retrieval_scores(query, gallery)
            rank = _retrieval_metrics_from_scores(scores)["rank"].astype(np.int64)  # type: ignore[index]
            m_all = np.ones_like(pid_ids, dtype=bool)
            if len(src) == 2 and tgt not in src:
                m_app = np.isin(pid_ids, _applicable_pid_ids_for_pair_to_target(src, tgt))
                m_non = ~m_app
                split_masks = {"all": m_all, "applicable": m_app, "non_applicable": m_non}
            else:
                split_masks = {"all": m_all}
            for split_name, mask in split_masks.items():
                met = _retrieval_metrics_from_rank_subset(rank, mask)
                rows.append(
                    {
                        "source": 0.0,
                        "target": 0.0,
                        "split": 0.0,
                        "n": float(met["n"]),
                        "recall_at_1": float(met["recall_at_1"]),
                        "recall_at_5": float(met["recall_at_5"]),
                        "mrr": float(met["mrr"]),
                        "random_recall_at_1": float(1.0 / rank.shape[0]),
                        "applicable_rate": float(np.mean(mask)),
                    }
                )
                rows[-1]["source"] = src  # type: ignore[assignment]
                rows[-1]["target"] = tgt  # type: ignore[assignment]
                rows[-1]["split"] = split_name  # type: ignore[assignment]

    with (out_dir / "compositional_easy_raw_retrieval_sanity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "target",
                "split",
                "n",
                "recall_at_1",
                "recall_at_5",
                "mrr",
                "random_recall_at_1",
                "applicable_rate",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Hard-slice summary for the rotated pair->heldout tasks.
    rot = [
        r
        for r in rows
        if (str(r["source"]), str(r["target"])) in {("23", "1"), ("13", "2"), ("12", "3")} and str(r["split"]) == "all"
    ]
    mean_pair_heldout_r1 = float(np.mean([float(r["recall_at_1"]) for r in rot]))
    random_r1 = float(rot[0]["random_recall_at_1"]) if rot else 0.0

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    order = [("12", "3"), ("13", "2"), ("23", "1"), ("123", "1"), ("123", "2"), ("123", "3")]
    vals = []
    labels = []
    for s, t in order:
        rr = [r for r in rows if str(r["source"]) == s and str(r["target"]) == t and str(r["split"]) == "all"]
        vals.append(float(rr[0]["recall_at_1"]))
        labels.append(f"{s}->{t}")
    ax.bar(np.arange(len(vals)), vals, color="#4c78a8")
    ax.axhline(random_r1, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="random R@1")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Recall@1")
    ax.set_title("Compositional-easy RAW retrieval sanity (exact instance)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _savefig(out_dir / "compositional_easy_raw_retrieval_sanity.png")

    # We only need a clear proof-of-solvability. Threshold is intentionally modest.
    assert mean_pair_heldout_r1 > max(10.0 * random_r1, 0.01), (mean_pair_heldout_r1, random_r1)


def test_dataset_difficulty_ladder_raw_retrieval():
    """
    Dataset-side difficulty ladder for RAW exact-instance retrieval.

    This isolates benchmarkability before rerunning full SSL comparisons. We track
    how pair->heldout retrieval changes as we move from the original single-atom
    strict generator to compositional variants of increasing difficulty.
    """
    out_dir = _ensure_plot_dir()

    ladder = [
        {
            "level": "L0",
            "name": "compositional_very_easy",
            "cfg": PIDDatasetConfig(
                d=32, m=8, sigma=0.02, seed=3901,
                rho_choices=(0.8,), hop_choices=(1,),
                deleakage_fit_samples=1024,
                composition_mode="multi_atom",
                active_atoms_per_sample=5,
                shared_backbone_gain=4.0,
                shared_backbone_tied_projection=True,
                synergy_deleak_lambda=0.25,
            ),
        },
        {
            "level": "L1",
            "name": "compositional_easy_plus",
            "cfg": PIDDatasetConfig(
                d=32, m=8, sigma=0.025, seed=3902,
                rho_choices=(0.5, 0.8), hop_choices=(1, 2),
                deleakage_fit_samples=1024,
                composition_mode="multi_atom",
                active_atoms_per_sample=4,
                shared_backbone_gain=3.2,
                shared_backbone_tied_projection=True,
                synergy_deleak_lambda=0.35,
            ),
        },
        {
            "level": "L2",
            "name": "compositional_easy",
            "cfg": PIDDatasetConfig(
                d=32, m=8, sigma=0.03, seed=3903,
                rho_choices=(0.5, 0.8), hop_choices=(1, 2),
                deleakage_fit_samples=1024,
                composition_mode="multi_atom",
                active_atoms_per_sample=4,
                shared_backbone_gain=2.5,
                shared_backbone_tied_projection=True,
                synergy_deleak_lambda=0.5,
            ),
        },
    ]

    source_map = {"12": ("x1", "x2"), "13": ("x1", "x3"), "23": ("x2", "x3"), "123": ("x1", "x2", "x3")}
    target_keys = {"1": "x1", "2": "x2", "3": "x3"}
    pair_heldout = {("12", "3"), ("13", "2"), ("23", "1")}

    rows: List[Dict[str, float]] = []
    group_rows: List[Dict[str, float]] = []
    for item in ladder:
        level = str(item["level"])
        name = str(item["name"])
        gen = PIDSar3DatasetGenerator(item["cfg"])  # type: ignore[arg-type]
        batch = _balanced_batch(gen, n_per_pid=180, shuffle_seed=959, return_aux=True)
        pid_ids = batch["pid_id"].astype(np.int64)
        X = _concat_raw(batch)
        parts = _split_modalities_from_concat(X)

        per_task: List[Dict[str, float]] = []
        for src, src_keys in source_map.items():
            query = _fused_query_from_parts(parts, src_keys)
            for tgt, tgt_key in target_keys.items():
                gallery = parts[tgt_key]
                rank = _retrieval_metrics_from_scores(_retrieval_scores(query, gallery))["rank"].astype(np.int64)  # type: ignore[index]
                task_key = (src, tgt)
                if task_key in pair_heldout:
                    split_masks = {
                        "all": np.ones_like(pid_ids, dtype=bool),
                        "applicable": np.isin(pid_ids, _applicable_pid_ids_for_pair_to_target(src, tgt)),
                    }
                    split_masks["non_applicable"] = ~split_masks["applicable"]
                else:
                    split_masks = {"all": np.ones_like(pid_ids, dtype=bool)}
                for split_name, mask in split_masks.items():
                    met = _retrieval_metrics_from_rank_subset(rank, mask)
                    row = {
                        "level": 0.0,
                        "setting": 0.0,
                        "source": 0.0,
                        "target": 0.0,
                        "split": 0.0,
                        "n": float(met["n"]),
                        "recall_at_1": float(met["recall_at_1"]),
                        "recall_at_5": float(met["recall_at_5"]),
                        "mrr": float(met["mrr"]),
                        "random_recall_at_1": float(1.0 / rank.shape[0]),
                        "sigma": float(item["cfg"].sigma),  # type: ignore[index]
                        "active_atoms_per_sample": float(item["cfg"].active_atoms_per_sample),  # type: ignore[index]
                        "shared_backbone_gain": float(item["cfg"].shared_backbone_gain),  # type: ignore[index]
                        "shared_backbone_tied_projection": float(1.0 if item["cfg"].shared_backbone_tied_projection else 0.0),  # type: ignore[index]
                        "synergy_deleak_lambda": float(item["cfg"].synergy_deleak_lambda),  # type: ignore[index]
                    }
                    row["level"] = level  # type: ignore[assignment]
                    row["setting"] = name  # type: ignore[assignment]
                    row["source"] = src  # type: ignore[assignment]
                    row["target"] = tgt  # type: ignore[assignment]
                    row["split"] = split_name  # type: ignore[assignment]
                    rows.append(row)
                    per_task.append(row)

        # grouped summaries for all-split rows
        all_rows = [r for r in per_task if str(r["split"]) == "all"]
        def _g(src: str, tgt: str) -> str:
            if len(src) == 2 and tgt not in src:
                return "pair_to_heldout"
            if len(src) == 2 and tgt in src:
                return "pair_to_member"
            if len(src) == 3:
                return "triple_to_target"
            return "other"
        for gname in ("pair_to_heldout", "pair_to_member", "triple_to_target"):
            xs = [r for r in all_rows if _g(str(r["source"]), str(r["target"])) == gname]
            if not xs:
                continue
            group_rows.append(
                {
                    "level": level,
                    "setting": name,
                    "group": gname,
                    "recall_at_1_mean": float(np.mean([float(r["recall_at_1"]) for r in xs])),
                    "recall_at_5_mean": float(np.mean([float(r["recall_at_5"]) for r in xs])),
                    "mrr_mean": float(np.mean([float(r["mrr"]) for r in xs])),
                    "random_recall_at_1": float(xs[0]["random_recall_at_1"]),
                }
            )

    with (out_dir / "dataset_difficulty_ladder_raw_retrieval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level", "setting", "source", "target", "split", "n", "recall_at_1", "recall_at_5", "mrr",
                "random_recall_at_1", "sigma", "active_atoms_per_sample", "shared_backbone_gain",
                "shared_backbone_tied_projection", "synergy_deleak_lambda",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with (out_dir / "dataset_difficulty_ladder_raw_retrieval_grouped.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["level", "setting", "group", "recall_at_1_mean", "recall_at_5_mean", "mrr_mean", "random_recall_at_1"]
        )
        writer.writeheader()
        for r in group_rows:
            writer.writerow(r)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8), sharex=True)
    order = [str(item["level"]) for item in ladder]
    labels = [f'{item["level"]}\\n{item["name"]}' for item in ladder]
    grouped_df = {(str(r["level"]), str(r["group"])): r for r in group_rows}
    for ax, group_name, color in zip(
        axes,
        ["pair_to_heldout", "pair_to_member", "triple_to_target"],
        ["#e45756", "#54a24b", "#4c78a8"],
    ):
        ys = [float(grouped_df[(lev, group_name)]["recall_at_1_mean"]) for lev in order]
        ax.plot(np.arange(len(order)), ys, marker="o", linewidth=2.0, color=color)
        rand = float(group_rows[0]["random_recall_at_1"])
        ax.axhline(rand, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(group_name.replace("_", " "))
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("RAW Recall@1")
    fig.suptitle("Dataset difficulty ladder (RAW exact-instance retrieval)", y=1.02)
    fig.tight_layout()
    _savefig(out_dir / "dataset_difficulty_ladder_raw_retrieval.png")

    # Sanity: the easy compositional level should be clearly solvable on pair->heldout.
    easy_pair = [r for r in group_rows if str(r["level"]) == "L2" and str(r["group"]) == "pair_to_heldout"][0]
    assert float(easy_pair["recall_at_1_mean"]) > 0.05


def test_analysis_bundle_four_models_compositional_very_easy():
    """
    Train the 4-model suite once on the compositional-very-easy dataset (L0) and
    export the main downstream analyses used in the results doc:
    (i) source->target kappa (5-fold), (ii) frozen retrieval, (iii) frozen-decoder
    reconstruction (5-fold, pair/triple sources).
    """
    out_dir = _ensure_plot_dir()
    data_cfg = _data_cfg_compositional_very_easy(seed=4001)
    ssl_gen = PIDSar3DatasetGenerator(data_cfg)
    enc_cfg = SSLEncoderConfig(input_dim=data_cfg.d, encoder_hidden_dim=96, representation_dim=48, projector_hidden_dim=96, projector_dim=48)
    train_cfg = SSLTrainConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=192,
        steps=180,
        temperature=0.2,
        device="cpu",
        seed=141,
        triangle_reg_weight=0.15,
        confu_pair_weight=0.5,
        confu_fused_weight=0.5,
    )

    unimodal_models, _ = _train_model_a_unimodal_sum_simclr(ssl_gen, enc_cfg, train_cfg)
    model_b, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "pairwise_simclr", "sum_3_pairwise_infonce")
    model_c, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "triangle_exact", "triangle_exact")
    model_d, _ = _train_trimodal_objective(ssl_gen, enc_cfg, train_cfg, "confu_style", "confu_style")

    model_labels = {
        "A": "A: 3x unimodal SimCLR",
        "B": "B: pairwise InfoNCE",
        "C": "C: TRIANGLE",
        "D": "D: ConFu",
    }
    source_map_full = {
        "1": ("x1",),
        "2": ("x2",),
        "3": ("x3",),
        "12": ("x1", "x2"),
        "13": ("x1", "x3"),
        "23": ("x2", "x3"),
        "123": ("x1", "x2", "x3"),
    }
    source_map_kappa = {"12": ("x1", "x2"), "13": ("x1", "x3"), "23": ("x2", "x3"), "123": ("x1", "x2", "x3")}
    source_map_recon = {"12": ("x1", "x2"), "13": ("x1", "x3"), "23": ("x2", "x3"), "123": ("x1", "x2", "x3")}
    target_keys = {"1": "x1", "2": "x2", "3": "x3"}

    # Shared probe set for retrieval + reconstruction (and also used for kappa folds).
    probe_gen = PIDSar3DatasetGenerator(_data_cfg_compositional_very_easy(seed=int(data_cfg.seed)))
    probe = _balanced_batch(probe_gen, n_per_pid=40, shuffle_seed=969, return_aux=True)
    pid_ids = probe["pid_id"].astype(np.int64)
    y_all_pid = pid_ids.copy()

    Xs = {
        "A": _concat_unimodal_frozen(unimodal_models, probe),
        "B": _concat_trimodal_frozen(model_b, probe),
        "C": _concat_trimodal_frozen(model_c, probe),
        "D": _concat_trimodal_frozen(model_d, probe),
    }

    # -------------------------------------------------------------------------
    # (i) Source->target kappa (5-fold)
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
    kappa_fold_rows: List[Dict[str, float]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros_like(y_all_pid), y_all_pid), start=1):
        batch_tr = _slice_batch(probe, tr_idx)
        batch_te = _slice_batch(probe, te_idx)
        for mkey, X in Xs.items():
            Xtr = X[tr_idx]
            Xte = X[te_idx]
            parts_tr = _split_modalities_from_concat(Xtr)
            parts_te = _split_modalities_from_concat(Xte)
            for src, src_keys in source_map_kappa.items():
                Xsrc_tr = _subset_concat(parts_tr, src_keys)
                Xsrc_te = _subset_concat(parts_te, src_keys)
                for tgt in ("1", "2", "3"):
                    metrics = _fit_binary_macro_f1_kappa_over_target_dims(
                        Xsrc_tr,
                        batch_tr[target_keys[tgt]].astype(np.float32),
                        Xsrc_te,
                        batch_te[target_keys[tgt]].astype(np.float32),
                    )
                    kappa_fold_rows.append(
                        {
                            "fold": float(fold_idx),
                            "model": 0.0,
                            "model_label": 0.0,
                            "source": 0.0,
                            "target": 0.0,
                            "macro_f1": float(metrics["macro_f1"]),
                            "macro_kappa": float(metrics["macro_kappa"]),
                            "n_target_dims": float(metrics["n_target_dims"]),
                        }
                    )
                    kappa_fold_rows[-1]["model"] = mkey  # type: ignore[assignment]
                    kappa_fold_rows[-1]["model_label"] = model_labels[mkey]  # type: ignore[assignment]
                    kappa_fold_rows[-1]["source"] = src  # type: ignore[assignment]
                    kappa_fold_rows[-1]["target"] = tgt  # type: ignore[assignment]

    kappa_prefix = "compositional_very_easy_source_to_target_four_models_5fold"
    with (out_dir / f"{kappa_prefix}_per_fold.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["fold", "model", "model_label", "source", "target", "macro_f1", "macro_kappa", "n_target_dims"]
        )
        writer.writeheader()
        for r in kappa_fold_rows:
            writer.writerow(r)

    grouped_k: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}
    for r in kappa_fold_rows:
        grouped_k.setdefault((str(r["model"]), str(r["source"]), str(r["target"])), []).append(r)
    kappa_summary_rows: List[Dict[str, float]] = []
    for (m, src, tgt), rows in grouped_k.items():
        f1_stats = _mean_ci95([float(r["macro_f1"]) for r in rows])
        k_stats = _mean_ci95([float(r["macro_kappa"]) for r in rows])
        row = {
            "model": 0.0, "model_label": 0.0, "source": 0.0, "target": 0.0,
            "n_folds": float(len(rows)),
            "macro_f1_mean": float(f1_stats["mean"]), "macro_f1_se": float(f1_stats["se"]),
            "macro_kappa_mean": float(k_stats["mean"]), "macro_kappa_se": float(k_stats["se"]),
        }
        row["model"] = m  # type: ignore[assignment]
        row["model_label"] = model_labels[m]  # type: ignore[assignment]
        row["source"] = src  # type: ignore[assignment]
        row["target"] = tgt  # type: ignore[assignment]
        kappa_summary_rows.append(row)
    kappa_summary_rows.sort(key=lambda r: (str(r["model"]), len(str(r["source"])), str(r["source"]), str(r["target"])))
    with (out_dir / f"{kappa_prefix}_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "model_label", "source", "target", "n_folds",
                "macro_f1_mean", "macro_f1_se", "macro_kappa_mean", "macro_kappa_se",
            ],
        )
        writer.writeheader()
        for r in kappa_summary_rows:
            writer.writerow(r)

    def _group_name(src: str, tgt: str) -> str:
        if len(src) == 1 and src == tgt:
            return "self_1to1"
        if len(src) == 1 and src != tgt:
            return "single_cross"
        if len(src) == 2 and tgt not in src:
            return "pair_to_heldout_target"
        if len(src) == 2 and tgt in src:
            return "pair_to_member_target"
        if len(src) == 3:
            return "triple_to_target"
        return "other"

    kappa_group_rows: List[Dict[str, float]] = []
    by_model_group: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for r in kappa_summary_rows:
        by_model_group.setdefault((str(r["model"]), _group_name(str(r["source"]), str(r["target"]))), []).append(r)
    for (m, g), rows in by_model_group.items():
        row = {
            "model": 0.0, "model_label": 0.0, "group": 0.0,
            "macro_f1_mean": float(np.mean([float(r["macro_f1_mean"]) for r in rows])),
            "macro_kappa_mean": float(np.mean([float(r["macro_kappa_mean"]) for r in rows])),
        }
        row["model"] = m  # type: ignore[assignment]
        row["model_label"] = model_labels[m]  # type: ignore[assignment]
        row["group"] = g  # type: ignore[assignment]
        kappa_group_rows.append(row)
    with (out_dir / f"{kappa_prefix}_grouped_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "model_label", "group", "macro_f1_mean", "macro_kappa_mean"])
        writer.writeheader()
        for r in kappa_group_rows:
            writer.writerow(r)

    # -------------------------------------------------------------------------
    # (ii) Frozen retrieval (single-run)
    # -------------------------------------------------------------------------
    retrieval_rows: List[Dict[str, float]] = []
    retrieval_strat_rows: List[Dict[str, float]] = []
    for mkey, X in Xs.items():
        parts = _split_modalities_from_concat(X)
        for src, src_keys in source_map_full.items():
            query = _fused_query_from_parts(parts, src_keys)
            for tgt, tgt_key in target_keys.items():
                gallery = parts[tgt_key]
                scores = _retrieval_scores(query, gallery)
                metrics = _retrieval_metrics_from_scores(scores)
                rank = metrics["rank"]  # type: ignore[index]
                row = {
                    "model": 0.0, "model_label": 0.0, "source": 0.0, "target": 0.0,
                    "recall_at_1": float(metrics["recall_at_1"][0]),
                    "recall_at_5": float(metrics["recall_at_5"][0]),
                    "mrr": float(metrics["mrr"][0]),
                    "n": float(rank.shape[0]),
                }
                row["model"] = mkey  # type: ignore[assignment]
                row["model_label"] = model_labels[mkey]  # type: ignore[assignment]
                row["source"] = src  # type: ignore[assignment]
                row["target"] = tgt  # type: ignore[assignment]
                retrieval_rows.append(row)
                for r in _retrieval_metrics_stratified(rank.astype(np.int64), pid_ids):
                    rr = dict(r)
                    rr.update({"model": mkey, "model_label": model_labels[mkey], "source": src, "target": tgt})
                    retrieval_strat_rows.append(rr)

    retrieval_prefix = "compositional_very_easy_retrieval_source_to_target_four_models"
    with (out_dir / f"{retrieval_prefix}_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "model_label", "source", "target", "recall_at_1", "recall_at_5", "mrr", "n"]
        )
        writer.writeheader()
        for r in retrieval_rows:
            writer.writerow(r)
    with (out_dir / f"{retrieval_prefix}_stratified.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "model_label", "source", "target", "scope", "label_id", "label_name", "n",
                "recall_at_1", "recall_at_5", "mrr",
            ],
        )
        writer.writeheader()
        for r in retrieval_strat_rows:
            writer.writerow(r)

    # -------------------------------------------------------------------------
    # (iii) Frozen-decoder reconstruction (5-fold; pair/triple sources)
    # -------------------------------------------------------------------------
    recon_fold_rows: List[Dict[str, float]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros_like(y_all_pid), y_all_pid), start=1):
        batch_tr = _slice_batch(probe, tr_idx)
        batch_te = _slice_batch(probe, te_idx)
        for mkey, X in Xs.items():
            Xtr = X[tr_idx]
            Xte = X[te_idx]
            parts_tr = _split_modalities_from_concat(Xtr)
            parts_te = _split_modalities_from_concat(Xte)
            for src, src_keys in source_map_recon.items():
                Xsrc_tr = _subset_concat(parts_tr, src_keys)
                Xsrc_te = _subset_concat(parts_te, src_keys)
                for tgt in ("1", "2", "3"):
                    Ytr = batch_tr[target_keys[tgt]].astype(np.float32)
                    Yte = batch_te[target_keys[tgt]].astype(np.float32)
                    for dec_key in ("ridge", "mlp"):
                        metrics = _fit_reconstruction_decoder_metrics(Xsrc_tr, Ytr, Xsrc_te, Yte, decoder=dec_key)
                        row = {
                            "fold": float(fold_idx),
                            "model": 0.0, "model_label": 0.0,
                            "decoder": 0.0, "decoder_label": 0.0,
                            "source": 0.0, "target": 0.0,
                            "macro_r2": float(metrics["macro_r2"]),
                            "macro_nrmse": float(metrics["macro_nrmse"]),
                            "n_target_dims": float(metrics["n_target_dims"]),
                        }
                        row["model"] = mkey  # type: ignore[assignment]
                        row["model_label"] = model_labels[mkey]  # type: ignore[assignment]
                        row["decoder"] = dec_key  # type: ignore[assignment]
                        row["decoder_label"] = "Ridge" if dec_key == "ridge" else "MLP"  # type: ignore[assignment]
                        row["source"] = src  # type: ignore[assignment]
                        row["target"] = tgt  # type: ignore[assignment]
                        recon_fold_rows.append(row)

    recon_prefix = "compositional_very_easy_source_to_target_reconstruction_four_models_5fold"
    with (out_dir / f"{recon_prefix}_per_fold.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold", "model", "model_label", "decoder", "decoder_label", "source", "target",
                "macro_r2", "macro_nrmse", "n_target_dims",
            ],
        )
        writer.writeheader()
        for r in recon_fold_rows:
            writer.writerow(r)

    grouped_r: Dict[Tuple[str, str, str, str], List[Dict[str, float]]] = {}
    for r in recon_fold_rows:
        grouped_r.setdefault((str(r["model"]), str(r["decoder"]), str(r["source"]), str(r["target"])), []).append(r)
    recon_summary_rows: List[Dict[str, float]] = []
    for (m, dec, src, tgt), rows in grouped_r.items():
        r2_stats = _mean_ci95([float(r["macro_r2"]) for r in rows])
        nr_stats = _mean_ci95([float(r["macro_nrmse"]) for r in rows])
        row = {
            "model": 0.0, "model_label": 0.0, "decoder": 0.0, "decoder_label": 0.0, "source": 0.0, "target": 0.0,
            "n_folds": float(len(rows)),
            "macro_r2_mean": float(r2_stats["mean"]), "macro_r2_se": float(r2_stats["se"]),
            "macro_nrmse_mean": float(nr_stats["mean"]), "macro_nrmse_se": float(nr_stats["se"]),
        }
        row["model"] = m  # type: ignore[assignment]
        row["model_label"] = model_labels[m]  # type: ignore[assignment]
        row["decoder"] = dec  # type: ignore[assignment]
        row["decoder_label"] = "Ridge" if dec == "ridge" else "MLP"  # type: ignore[assignment]
        row["source"] = src  # type: ignore[assignment]
        row["target"] = tgt  # type: ignore[assignment]
        recon_summary_rows.append(row)
    recon_summary_rows.sort(key=lambda r: (str(r["decoder"]), str(r["model"]), len(str(r["source"])), str(r["source"]), str(r["target"])))
    with (out_dir / f"{recon_prefix}_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "model_label", "decoder", "decoder_label", "source", "target", "n_folds",
                "macro_r2_mean", "macro_r2_se", "macro_nrmse_mean", "macro_nrmse_se",
            ],
        )
        writer.writeheader()
        for r in recon_summary_rows:
            writer.writerow(r)

    recon_group_rows: List[Dict[str, float]] = []
    by_model_group_r: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}
    for r in recon_summary_rows:
        by_model_group_r.setdefault((str(r["decoder"]), str(r["model"]), _group_name(str(r["source"]), str(r["target"]))), []).append(r)
    for (dec, m, g), rows in by_model_group_r.items():
        row = {
            "decoder": 0.0, "decoder_label": 0.0, "model": 0.0, "model_label": 0.0, "group": 0.0,
            "macro_r2_mean": float(np.mean([float(r["macro_r2_mean"]) for r in rows])),
            "macro_nrmse_mean": float(np.mean([float(r["macro_nrmse_mean"]) for r in rows])),
        }
        row["decoder"] = dec  # type: ignore[assignment]
        row["decoder_label"] = "Ridge" if dec == "ridge" else "MLP"  # type: ignore[assignment]
        row["model"] = m  # type: ignore[assignment]
        row["model_label"] = model_labels[m]  # type: ignore[assignment]
        row["group"] = g  # type: ignore[assignment]
        recon_group_rows.append(row)
    with (out_dir / f"{recon_prefix}_grouped_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["decoder", "decoder_label", "model", "model_label", "group", "macro_r2_mean", "macro_nrmse_mean"]
        )
        writer.writeheader()
        for r in recon_group_rows:
            writer.writerow(r)

    assert len(kappa_summary_rows) == 4 * 4 * 3
    assert len(retrieval_rows) == 4 * 7 * 3
    assert len(recon_summary_rows) == 2 * 4 * 4 * 3
