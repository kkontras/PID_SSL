"""Evaluation metrics: alignment, uniformity, effective rank, gradient cosine."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def alignment(z_i: np.ndarray, z_j: np.ndarray) -> float:
    """
    Mean negative squared distance between positive pairs (alignment metric).
    z_i, z_j: (N, d), L2-normalized.
    Higher = more aligned.
    """
    diff = z_i - z_j
    return float(-np.mean(np.sum(diff ** 2, axis=-1)))


def uniformity(z: np.ndarray, t: float = 2.0) -> float:
    """
    Log-average Gaussian pairwise kernel (uniformity metric).
    z: (N, d), L2-normalized.
    Lower = more uniform on the hypersphere.
    """
    # Pairwise squared distances
    sq_dists = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)  # (N, N)
    return float(np.log(np.mean(np.exp(-t * sq_dists)) + 1e-8))


def effective_rank(z: np.ndarray) -> float:
    """
    Effective rank of representation matrix Z (N, d).
    erank = exp(entropy of normalized singular values).
    Higher = more spread across dimensions (less collapse).
    """
    # SVD
    s = np.linalg.svd(z, compute_uv=False)
    s = s[s > 1e-12]
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    return float(np.exp(entropy))


def positive_similarity(z_i: np.ndarray, z_j: np.ndarray) -> float:
    """Mean cosine similarity for positive pairs (L2-normalized inputs)."""
    return float(np.mean(np.sum(z_i * z_j, axis=-1)))


def negative_similarity(z_i: np.ndarray, z_j: np.ndarray, n_samples: int = 1000) -> float:
    """Mean cosine similarity for random non-matching pairs."""
    N = z_i.shape[0]
    rng = np.random.default_rng(42)
    idx_i = rng.integers(0, N, size=n_samples)
    idx_j = rng.integers(0, N, size=n_samples)
    # Avoid matching pairs
    match = idx_i == idx_j
    idx_j[match] = (idx_j[match] + 1) % N
    return float(np.mean(np.sum(z_i[idx_i] * z_j[idx_j], axis=-1)))


def representation_diagnostics(z1: np.ndarray, z2: np.ndarray, z3: np.ndarray) -> Dict[str, float]:
    """Compute a battery of representation diagnostics."""
    return {
        "align_12": alignment(z1, z2),
        "align_13": alignment(z1, z3),
        "align_23": alignment(z2, z3),
        "unif_z1": uniformity(z1),
        "unif_z2": uniformity(z2),
        "unif_z3": uniformity(z3),
        "erank_z1": effective_rank(z1),
        "erank_z2": effective_rank(z2),
        "erank_z3": effective_rank(z3),
        "pos_sim_12": positive_similarity(z1, z2),
        "pos_sim_13": positive_similarity(z1, z3),
        "pos_sim_23": positive_similarity(z2, z3),
        "neg_sim_12": negative_similarity(z1, z2),
        "neg_sim_13": negative_similarity(z1, z3),
        "neg_sim_23": negative_similarity(z2, z3),
    }
