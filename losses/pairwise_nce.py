"""Pairwise symmetric InfoNCE (NT-Xent variant) for cross-modal contrastive learning."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE between two L2-normalized representations.
    z_i, z_j: (N, d_z), already L2-normalized.
    Positive pairs are on the diagonal (i.e., z_i[n] and z_j[n] are a positive pair).
    """
    N = z_i.shape[0]
    labels = torch.arange(N, device=z_i.device)

    sim = torch.mm(z_i, z_j.t()) / temperature  # (N, N)

    loss_ij = F.cross_entropy(sim, labels)
    loss_ji = F.cross_entropy(sim.t(), labels)
    return (loss_ij + loss_ji) / 2.0


def pairwise_infonce_loss(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
                          temperature: float = 0.07) -> torch.Tensor:
    """Sum of symmetric InfoNCE over all 3 pairs."""
    return (info_nce_loss(z1, z2, temperature) +
            info_nce_loss(z1, z3, temperature) +
            info_nce_loss(z2, z3, temperature))
