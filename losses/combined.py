"""Combined loss: contrastive + classification cross-entropy."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from losses.pairwise_nce import pairwise_infonce_loss
from losses.simclr import simclr_loss_per_modality
from losses.triangle import triangle_contrastive_loss
from losses.confu import confu_loss
from losses.comm import comm_loss, infmask_loss
from losses.masked_pred import masked_raw_loss, masked_emb_loss


def combined_loss(
    method: str,
    z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    fused: Optional[Dict[str, torch.Tensor]] = None,
    # SimCLR augmented views
    z1_a: Optional[torch.Tensor] = None, z1_b: Optional[torch.Tensor] = None,
    z2_a: Optional[torch.Tensor] = None, z2_b: Optional[torch.Tensor] = None,
    z3_a: Optional[torch.Tensor] = None, z3_b: Optional[torch.Tensor] = None,
    zf_a: Optional[torch.Tensor] = None, zf_b: Optional[torch.Tensor] = None,
    zf_a_masked_list: Optional[Tuple[torch.Tensor, ...]] = None,
    zf_b_masked_list: Optional[Tuple[torch.Tensor, ...]] = None,
    temperature: float = 0.07,
    lambda_contr: float = 0.1,
    lambda_mask: float = 1.0,
    # method-specific
    triangle_alpha: float = 0.0,
    confu_pair_weight: float = 1.0,
    confu_fuse_weight: float = 1.0,
    # masked prediction
    masked_out: Optional[Dict] = None,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    x3: Optional[torch.Tensor] = None,
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
    mask3: Optional[torch.Tensor] = None,
    masked_emb_var_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Unified loss dispatcher.
    method: one of 'none', 'simclr', 'pairwise_nce', 'triangle', 'confu', 'comm', 'infmask'
    Returns: (total_loss, metrics_dict)
    """
    metrics: Dict[str, float] = {}

    # Classification loss
    ce_loss = torch.tensor(0.0, device=z1.device)
    if logits is not None and targets is not None:
        ce_loss = F.cross_entropy(logits, targets)
        metrics["ce_loss"] = float(ce_loss.detach().cpu())

    # Contrastive loss
    contr_loss = torch.tensor(0.0, device=z1.device)
    if method == "none":
        pass
    elif method == "simclr":
        assert z1_a is not None and z1_b is not None
        contr_loss = simclr_loss_per_modality(z1_a, z1_b, z2_a, z2_b, z3_a, z3_b, temperature)
    elif method == "pairwise_nce":
        contr_loss = pairwise_infonce_loss(z1, z2, z3, temperature)
    elif method == "triangle":
        contr_loss = triangle_contrastive_loss(z1, z2, z3, temperature, triangle_alpha)
    elif method == "confu":
        assert fused is not None
        contr_loss = confu_loss(z1, z2, z3, fused, temperature, confu_pair_weight, confu_fuse_weight)
    elif method == "comm":
        assert all(t is not None for t in (z1_a, z2_a, z3_a, z1_b, z2_b, z3_b, zf_a, zf_b))
        contr_loss = comm_loss(z1_a, z2_a, z3_a, z1_b, z2_b, z3_b, zf_a, zf_b, temperature)
    elif method == "infmask":
        assert all(t is not None for t in (z1_a, z2_a, z3_a, z1_b, z2_b, z3_b, zf_a, zf_b))
        assert zf_a_masked_list is not None and zf_b_masked_list is not None
        contr_loss = infmask_loss(
            z1_a, z2_a, z3_a,
            z1_b, z2_b, z3_b,
            zf_a, zf_b,
            zf_a_masked_list, zf_b_masked_list,
            temperature=temperature,
            lambda_mask=lambda_mask,
        )
    elif method == "masked_raw":
        assert masked_out is not None and x1 is not None
        contr_loss = masked_raw_loss(
            x1, x2, x3,
            masked_out["z1_masked"], masked_out["z2_masked"], masked_out["z3_masked"],
            *masked_out["raw_decs"],
            mask1, mask2, mask3,
        )
    elif method == "masked_emb":
        assert masked_out is not None
        contr_loss = masked_emb_loss(
            masked_out["h1_masked"], masked_out["h2_masked"], masked_out["h3_masked"],
            masked_out["h1_teacher"], masked_out["h2_teacher"], masked_out["h3_teacher"],
            *masked_out["emb_preds"],
            variance_weight=masked_emb_var_weight,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics["contr_loss"] = float(contr_loss.detach().cpu())

    total = ce_loss + lambda_contr * contr_loss
    metrics["total_loss"] = float(total.detach().cpu())
    return total, metrics
