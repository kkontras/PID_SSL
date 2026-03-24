"""Projection heads for contrastive learning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head: d_model -> d_hidden -> d_z, with L2 normalization.
    Used for contrastive learning.
    """

    def __init__(self, d_model: int, d_hidden: int, d_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_z),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) -> z: (B, d_z), L2-normalized."""
        z = self.net(h)
        return F.normalize(z, dim=-1)
