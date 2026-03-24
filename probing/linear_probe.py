"""Linear probe: freeze encoder representations and train a linear classifier."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: str = "cpu",
    seed: int = 0,
) -> Dict[str, object]:
    """
    Train a single linear layer on top of frozen representations.
    Returns dict with validation accuracy and training accuracy.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_va = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_va = torch.from_numpy(y_val.astype(np.int64)).to(device)

    d = X_tr.shape[1]
    probe = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    N = X_tr.shape[0]
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        idx = rng.permutation(N)
        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            x_b = X_tr[batch_idx]
            y_b = y_tr[batch_idx]
            logits = probe(x_b)
            loss = F.cross_entropy(logits, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            train_logits = probe(X_tr)
            val_logits = probe(X_va)
            train_loss = F.cross_entropy(train_logits, y_tr).item()
            val_loss = F.cross_entropy(val_logits, y_va).item()
            train_acc = (train_logits.argmax(dim=-1) == y_tr).float().mean().item()
            val_acc = (val_logits.argmax(dim=-1) == y_va).float().mean().item()
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )
        probe.train()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        train_acc = (probe(X_tr).argmax(dim=-1) == y_tr).float().mean().item()
        val_acc = (probe(X_va).argmax(dim=-1) == y_va).float().mean().item()

    return {"train_acc": train_acc, "val_acc": val_acc, "history": history}
