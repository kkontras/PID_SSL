"""Per-atom accuracy computation for composite labels."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from probing.linear_probe import train_linear_probe


def decode_sub_labels(labels: np.ndarray, Q: int, n_atoms: int) -> Dict[str, np.ndarray]:
    """
    Decode composite labels Y = sum_k y_k * Q^k into per-atom digits.
    Returns dict: atom_0, atom_1, ... -> (N,) arrays of integers in {0,...,Q-1}.
    """
    result = {}
    remaining = labels.copy()
    for k in range(n_atoms):
        result[f"atom_{k}"] = remaining % Q
        remaining = remaining // Q
    return result


def per_atom_accuracy_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    Q: int,
    n_atoms: int,
) -> Dict[str, float]:
    """
    Given full classification logits and targets, compute per-atom accuracy
    by decoding the predicted composite label into digits.
    """
    preds = logits.argmax(axis=-1)
    pred_digits = decode_sub_labels(preds, Q, n_atoms)
    true_digits = decode_sub_labels(targets, Q, n_atoms)
    return {
        k: float(np.mean(pred_digits[k] == true_digits[k]))
        for k in pred_digits
    }


def probe_per_atom(
    X_train: np.ndarray,
    X_val: np.ndarray,
    sub_labels_train: Dict[str, np.ndarray],
    sub_labels_val: Dict[str, np.ndarray],
    Q: int,
    atom_names: List[str],
    epochs: int = 100,
    lr: float = 1e-2,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train separate linear probes for each atom's sub-label.
    Returns dict: atom_name -> val_acc.
    """
    results = {}
    for atom in atom_names:
        y_tr = sub_labels_train[f"sub_{atom}"]
        y_va = sub_labels_val[f"sub_{atom}"]
        res = train_linear_probe(
            X_train, y_tr, X_val, y_va,
            n_classes=Q, epochs=epochs, lr=lr, device=device,
        )
        results[atom] = res["val_acc"]
    return results
