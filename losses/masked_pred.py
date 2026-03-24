"""Masked prediction SSL losses: masked_raw (MAE-style) and masked_emb (data2vec-style)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from models.decoder import RawDecoder, EmbPredictor


def random_feature_mask(x: torch.Tensor, mask_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (x_masked, mask) where mask is bool (True = masked position).
    Masked positions set to 0.0. Independent random mask per sample.
    x: (B, D)
    """
    B, D = x.shape
    n_mask = int(D * mask_ratio)
    mask = torch.zeros(B, D, dtype=torch.bool, device=x.device)
    for i in range(B):
        idx = torch.randperm(D, device=x.device)[:n_mask]
        mask[i, idx] = True
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def masked_raw_loss(
    x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
    z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
    dec1: RawDecoder, dec2: RawDecoder, dec3: RawDecoder,
    mask1: torch.Tensor, mask2: torch.Tensor, mask3: torch.Tensor,
) -> torch.Tensor:
    """
    MAE-style: predict original raw features at masked positions.
    z_i comes from masked input; z_j,z_k come from full (unmasked) inputs.
    """
    pred1 = dec1(z1, z2, z3)
    pred2 = dec2(z2, z1, z3)
    pred3 = dec3(z3, z1, z2)
    loss1 = F.mse_loss(pred1[mask1], x1[mask1])
    loss2 = F.mse_loss(pred2[mask2], x2[mask2])
    loss3 = F.mse_loss(pred3[mask3], x3[mask3])
    return (loss1 + loss2 + loss3) / 3


def masked_emb_loss(
    h1_masked: torch.Tensor, h2_masked: torch.Tensor, h3_masked: torch.Tensor,
    h1_teacher: torch.Tensor, h2_teacher: torch.Tensor, h3_teacher: torch.Tensor,
    pred1: EmbPredictor, pred2: EmbPredictor, pred3: EmbPredictor,
    variance_weight: float = 1.0,
) -> torch.Tensor:
    """
    Teacher-student masked prediction on encoder features with a variance floor.
    Student uses masked inputs; teacher uses EMA full inputs with stop-gradient.
    """
    t1 = h1_teacher.detach()
    t2 = h2_teacher.detach()
    t3 = h3_teacher.detach()
    p1 = pred1(h1_masked)
    p2 = pred2(h2_masked)
    p3 = pred3(h3_masked)
    reg_loss = (F.mse_loss(p1, t1) + F.mse_loss(p2, t2) + F.mse_loss(p3, t3)) / 3

    def _variance_loss(h: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(h.var(dim=0, unbiased=False) + 1e-4)
        return torch.relu(1.0 - std).mean()

    var_loss = (_variance_loss(h1_masked) + _variance_loss(h2_masked) + _variance_loss(h3_masked)) / 3
    return reg_loss + variance_weight * var_loss
