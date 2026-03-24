"""Decoder heads for masked prediction SSL methods."""
from __future__ import annotations

import torch
import torch.nn as nn


class RawDecoder(nn.Module):
    """
    Predicts raw features of one modality from:
      - z_self:   (B, d_z)  — encoder output of the masked modality
      - z_other1: (B, d_z)  — encoder output of modality 2
      - z_other2: (B, d_z)  — encoder output of modality 3
    Output: (B, D) — predicted raw features
    """

    def __init__(self, d_z: int, hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_z, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, z_self: torch.Tensor, z_other1: torch.Tensor, z_other2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_self, z_other1, z_other2], dim=-1))


class EmbPredictor(nn.Module):
    """
    Predicts clean encoder features for one modality from the masked student view.
    """

    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
