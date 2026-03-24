"""Per-modality SimCLR (intra-modal contrastive) loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Standard NT-Xent loss for paired views (z_a[i], z_b[i]) are positives.
    z_a, z_b: (N, d_z), L2-normalized.
    """
    N = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)  # (2N, d)
    sim = torch.mm(z, z.t()) / temperature  # (2N, 2N)

    # Mask self-similarities
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive indices: (i -> i+N, i+N -> i)
    targets = torch.arange(2 * N, device=z.device)
    targets[:N] += N
    targets[N:] -= N

    return F.cross_entropy(sim, targets)


def simclr_loss_per_modality(z1_a: torch.Tensor, z1_b: torch.Tensor,
                              z2_a: torch.Tensor, z2_b: torch.Tensor,
                              z3_a: torch.Tensor, z3_b: torch.Tensor,
                              temperature: float = 0.07) -> torch.Tensor:
    """
    Per-modality SimCLR: independently apply NT-Xent within each modality
    (two augmented views of the same sample are positives).
    All inputs L2-normalized.
    """
    l1 = nt_xent(z1_a, z1_b, temperature)
    l2 = nt_xent(z2_a, z2_b, temperature)
    l3 = nt_xent(z3_a, z3_b, temperature)
    return (l1 + l2 + l3) / 3.0
