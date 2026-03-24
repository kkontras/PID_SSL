"""TRIANGLE area-based trimodal contrastive loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def triangle_area(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
    """
    Triangle area for each triplet (z1[i], z2[i], z3[i]).
    Inputs L2-normalized: (N, d_z).
    Returns: (N,) areas.
    """
    u = z1 - z2   # (N, d)
    v = z1 - z3   # (N, d)
    uu = (u * u).sum(dim=-1)
    vv = (v * v).sum(dim=-1)
    uv = (u * v).sum(dim=-1)
    area_sq = torch.clamp(uu * vv - uv ** 2, min=1e-8)
    return 0.5 * torch.sqrt(area_sq)


def _triangle_logits_matrix(za: torch.Tensor, zb: torch.Tensor, zc: torch.Tensor,
                             temperature: float) -> torch.Tensor:
    """
    Compute (N, N) logit matrix where logits[i, j] = -A(za[j], zb[i], zc[i]) / tau.
    Vary za (the "anchor/target" modality) while keeping zb[i], zc[i] fixed.
    Diagonal = positives (same sample).
    """
    N = za.shape[0]
    # za: vary over j (candidate). zb, zc: fixed at i.
    za_j = za.unsqueeze(0)   # (1, N, d)
    zb_i = zb.unsqueeze(1)   # (N, 1, d)
    zc_i = zc.unsqueeze(1)   # (N, 1, d)
    u = za_j - zb_i          # (N, N, d)
    v = za_j - zc_i          # (N, N, d)
    uu = (u * u).sum(dim=-1)
    vv = (v * v).sum(dim=-1)
    uv = (u * v).sum(dim=-1)
    area = 0.5 * torch.sqrt(torch.clamp(uu * vv - uv ** 2, min=1e-8))  # (N, N)
    return -area / temperature  # (N, N), higher = more similar


def triangle_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
                               temperature: float = 0.07,
                               cosine_reg_alpha: float = 0.0) -> torch.Tensor:
    """
    Symmetrized TRIANGLE loss: for each anchor modality, create logit matrix
    by varying that modality and keeping the other two fixed.
    Optional cosine regularization to ensure some pairwise alignment.
    """
    N = z1.shape[0]
    labels = torch.arange(N, device=z1.device)
    total_loss = 0.0

    # Vary z1 (target), fix z2 and z3 as the paired modalities
    logits1 = _triangle_logits_matrix(z1, z2, z3, temperature)
    total_loss = total_loss + F.cross_entropy(logits1, labels)

    # Vary z2, fix z1 and z3
    logits2 = _triangle_logits_matrix(z2, z1, z3, temperature)
    total_loss = total_loss + F.cross_entropy(logits2, labels)

    # Vary z3, fix z1 and z2
    logits3 = _triangle_logits_matrix(z3, z1, z2, temperature)
    total_loss = total_loss + F.cross_entropy(logits3, labels)

    loss = total_loss / 3.0

    if cosine_reg_alpha > 0.0:
        cos12 = (z1 * z2).sum(dim=-1).mean()
        cos13 = (z1 * z3).sum(dim=-1).mean()
        cos23 = (z2 * z3).sum(dim=-1).mean()
        reg = (cos12 + cos13 + cos23) / 3.0
        loss = loss - cosine_reg_alpha * reg

    return loss
