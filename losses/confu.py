"""ConFu: Contrastive Fusion loss — pairwise + fusion-to-third InfoNCE terms."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from losses.pairwise_nce import info_nce_loss


def confu_loss(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
               fused: Dict[str, torch.Tensor],
               temperature: float = 0.07,
               pair_weight: float = 1.0,
               fuse_weight: float = 1.0) -> torch.Tensor:
    """
    ConFu loss = pair_weight * pairwise_InfoNCE + fuse_weight * fusion_InfoNCE.

    fused: dict with keys 'f12', 'f13', 'f23' — L2-normalized fused representations.
      f12 is used to align against z3
      f13 is used to align against z2
      f23 is used to align against z1

    All inputs L2-normalized.
    """
    # Pairwise terms
    l12 = info_nce_loss(z1, z2, temperature)
    l13 = info_nce_loss(z1, z3, temperature)
    l23 = info_nce_loss(z2, z3, temperature)
    pair_loss = (l12 + l13 + l23) / 3.0

    # Fusion-to-third terms: fused pair predicts the held-out modality
    lf12_3 = info_nce_loss(fused["f12"], z3, temperature)  # fusion(1,2) vs z3
    lf13_2 = info_nce_loss(fused["f13"], z2, temperature)  # fusion(1,3) vs z2
    lf23_1 = info_nce_loss(fused["f23"], z1, temperature)  # fusion(2,3) vs z1
    fuse_loss = (lf12_3 + lf13_2 + lf23_1) / 3.0

    return pair_weight * pair_loss + fuse_weight * fuse_loss
