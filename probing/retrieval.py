"""Label-aware cross-modal retrieval metrics for V3 frozen representations."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / denom


def pair_query_rep(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _l2_normalize(a + b)


def _recall_at_k(rank: np.ndarray, k: int) -> float:
    return float(np.mean(rank <= k))


def _mrr(rank: np.ndarray) -> float:
    return float(np.mean(1.0 / rank))


def label_aware_retrieval_metrics(query: np.ndarray, gallery: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray) -> Dict[str, float]:
    scores = _l2_normalize(query) @ _l2_normalize(gallery).T
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(query.shape[0], dtype=np.int64)
    for i in range(query.shape[0]):
        positives = gallery_labels[order[i]] == query_labels[i]
        pos_idx = int(np.argmax(positives))
        ranks[i] = pos_idx + 1 if positives[pos_idx] else gallery.shape[0] + 1

    label_counts = {int(lbl): int(np.sum(gallery_labels == lbl)) for lbl in np.unique(gallery_labels)}
    chance_r1 = float(np.mean([label_counts[int(lbl)] / max(len(gallery_labels), 1) for lbl in query_labels]))
    return {
        "r_at_1": _recall_at_k(ranks, 1),
        "r_at_5": _recall_at_k(ranks, min(5, gallery.shape[0])),
        "mrr": _mrr(ranks),
        "chance_r1": chance_r1,
    }


def train_linear_retrieval_adapter(
    query_train: np.ndarray,
    gallery_train: np.ndarray,
    labels_train: np.ndarray,
    query_test: np.ndarray,
    device: str = "cpu",
    epochs: int = 80,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_tr = torch.from_numpy(query_train.astype(np.float32)).to(device)
    g_tr = torch.from_numpy(gallery_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(labels_train.astype(np.int64)).to(device)
    x_te = torch.from_numpy(query_test.astype(np.float32)).to(device)

    d_in = x_tr.shape[1]
    d_out = g_tr.shape[1]
    adapter = nn.Linear(d_in, d_out, bias=False).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = min(256, x_tr.shape[0])
    for _ in range(epochs):
        perm = rng.permutation(x_tr.shape[0])
        for start in range(0, x_tr.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            q_b = x_tr[idx]
            y_b = y_tr[idx]
            pred = F.normalize(adapter(q_b), dim=-1)
            target = F.normalize(g_tr[idx], dim=-1)
            loss_align = 1.0 - torch.sum(pred * target, dim=-1).mean()
            logits = pred @ F.normalize(g_tr, dim=-1).T
            loss_ce = F.cross_entropy(logits, y_b)
            loss = loss_align + 0.5 * loss_ce
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        return F.normalize(adapter(x_te), dim=-1).cpu().numpy()


def evaluate_all_atom_retrieval(
    reps_train: Dict[str, np.ndarray],
    reps_test: Dict[str, np.ndarray],
    atom_labels_train: Dict[str, np.ndarray],
    atom_labels_test: Dict[str, np.ndarray],
    method: str,
    device: str = "cpu",
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    query_specs: List[Tuple[str, np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]] = [
        ("x1_to_x3", reps_train["z1"], reps_test["z1"], "x3", reps_train["z3"], reps_test["z3"]),
        ("x2_to_x3", reps_train["z2"], reps_test["z2"], "x3", reps_train["z3"], reps_test["z3"]),
        ("x12_to_x3", reps_train["f12"] if method == "confu" and "f12" in reps_train else pair_query_rep(reps_train["z1"], reps_train["z2"]), reps_test["f12"] if method == "confu" and "f12" in reps_test else pair_query_rep(reps_test["z1"], reps_test["z2"]), "x3", reps_train["z3"], reps_test["z3"]),
        ("x1_to_x2", reps_train["z1"], reps_test["z1"], "x2", reps_train["z2"], reps_test["z2"]),
        ("x3_to_x2", reps_train["z3"], reps_test["z3"], "x2", reps_train["z2"], reps_test["z2"]),
        ("x13_to_x2", reps_train["f13"] if method == "confu" and "f13" in reps_train else pair_query_rep(reps_train["z1"], reps_train["z3"]), reps_test["f13"] if method == "confu" and "f13" in reps_test else pair_query_rep(reps_test["z1"], reps_test["z3"]), "x2", reps_train["z2"], reps_test["z2"]),
        ("x2_to_x1", reps_train["z2"], reps_test["z2"], "x1", reps_train["z1"], reps_test["z1"]),
        ("x3_to_x1", reps_train["z3"], reps_test["z3"], "x1", reps_train["z1"], reps_test["z1"]),
        ("x23_to_x1", reps_train["f23"] if method == "confu" and "f23" in reps_train else pair_query_rep(reps_train["z2"], reps_train["z3"]), reps_test["f23"] if method == "confu" and "f23" in reps_test else pair_query_rep(reps_test["z2"], reps_test["z3"]), "x1", reps_train["z1"], reps_test["z1"]),
    ]

    for atom_name, labels_test in atom_labels_test.items():
        labels_train = atom_labels_train[atom_name]
        for query_name, query_train, query_test, target_name, gallery_train, gallery_test in query_specs:
            zero_metrics = label_aware_retrieval_metrics(query_test, gallery_test, labels_test, labels_test)
            rows.append(
                {
                    "atom": atom_name,
                    "query": query_name,
                    "target": target_name,
                    "mode": "zero_shot",
                    **zero_metrics,
                }
            )
            adapted_query_test = train_linear_retrieval_adapter(
                query_train=query_train,
                gallery_train=gallery_train,
                labels_train=labels_train,
                query_test=query_test,
                device=device,
            )
            adapted_metrics = label_aware_retrieval_metrics(adapted_query_test, gallery_test, labels_test, labels_test)
            rows.append(
                {
                    "atom": atom_name,
                    "query": query_name,
                    "target": target_name,
                    "mode": "linear_adapter",
                    **adapted_metrics,
                }
            )
    return rows
