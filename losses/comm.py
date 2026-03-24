"""CoMM and InfMasking losses for multimodal contrastive pretraining."""
from __future__ import annotations

from typing import Sequence

import torch

from losses.pairwise_nce import info_nce_loss


def comm_loss(
    z1_a: torch.Tensor,
    z2_a: torch.Tensor,
    z3_a: torch.Tensor,
    z1_b: torch.Tensor,
    z2_b: torch.Tensor,
    z3_b: torch.Tensor,
    zf_a: torch.Tensor,
    zf_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Paper-style CoMM loss: fused-fused plus unimodal-fused alignment."""
    fused_term = info_nce_loss(zf_a, zf_b, temperature)
    modality_terms = [
        0.5 * (info_nce_loss(z1_a, zf_b, temperature) + info_nce_loss(z1_b, zf_a, temperature)),
        0.5 * (info_nce_loss(z2_a, zf_b, temperature) + info_nce_loss(z2_b, zf_a, temperature)),
        0.5 * (info_nce_loss(z3_a, zf_b, temperature) + info_nce_loss(z3_b, zf_a, temperature)),
    ]
    return fused_term + sum(modality_terms)


def infmask_loss(
    z1_a: torch.Tensor,
    z2_a: torch.Tensor,
    z3_a: torch.Tensor,
    z1_b: torch.Tensor,
    z2_b: torch.Tensor,
    z3_b: torch.Tensor,
    zf_a: torch.Tensor,
    zf_b: torch.Tensor,
    zf_a_masked_list: Sequence[torch.Tensor],
    zf_b_masked_list: Sequence[torch.Tensor],
    temperature: float = 0.07,
    lambda_mask: float = 1.0,
) -> torch.Tensor:
    """InfMasking: CoMM plus masked-fused to clean-fused alignment."""
    base_loss = comm_loss(z1_a, z2_a, z3_a, z1_b, z2_b, z3_b, zf_a, zf_b, temperature)
    if not zf_a_masked_list or not zf_b_masked_list:
        return base_loss

    n = min(len(zf_a_masked_list), len(zf_b_masked_list))
    mask_terms = []
    for idx in range(n):
        mask_terms.append(
            info_nce_loss(zf_a_masked_list[idx], zf_a, temperature) +
            info_nce_loss(zf_b_masked_list[idx], zf_b, temperature)
        )
    mask_term = sum(mask_terms) / float(n)
    return base_loss + lambda_mask * mask_term
