"""FusionMLP for ConFu-style higher-order contrastive learning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionMLP(nn.Module):
    """
    Fuses two modality projections z_i, z_j into a combined representation.
    Output is L2-normalized. Uses dropout to prevent shortcut learning.
    """

    def __init__(self, d_z: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(2 * d_z, d_z),
            nn.ReLU(),
            nn.Linear(d_z, d_z),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, mask_prob: float = 0.0) -> torch.Tensor:
        """
        z_i, z_j: (B, d_z), L2-normalized
        mask_prob: probability of zeroing out one of the inputs during training
        returns: (B, d_z), L2-normalized
        """
        if self.training and mask_prob > 0.0:
            # Randomly mask one input per sample
            mask = (torch.rand(z_i.shape[0], 1, device=z_i.device) > mask_prob).float()
            # Mask z_i or z_j alternately (flip for variety)
            z_i = z_i * mask
            z_j = z_j * (1 - mask + mask)  # keep z_j always for now

        z_i = self.dropout(z_i)
        z_j = self.dropout(z_j)
        fused = self.net(torch.cat([z_i, z_j], dim=-1))
        return F.normalize(fused, dim=-1)
