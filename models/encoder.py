"""NodeEncoder: Transformer-based encoder for a single synthetic modality node."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import torch
import torch.nn as nn


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
)


@dataclass
class NodeEncoderConfig:
    d_input: int = 24          # raw feature dimension per node
    d_model: int = 64          # Transformer hidden dim
    n_heads: int = 4           # attention heads
    n_layers: int = 2          # Transformer layers
    d_ff: int = 128            # FFN inner dim
    n_patches: int = 4         # number of patches (d_input must be divisible by n_patches)
    dropout: float = 0.0


class NodeEncoder(nn.Module):
    """
    Encodes a single node's feature vector via:
      1. Split into T patches, linearly project each to d_model
      2. Prepend a learnable CLS token
      3. Add learnable positional embeddings
      4. L Transformer encoder layers
      5. Return CLS token output as representation h ∈ R^d_model
    """

    def __init__(self, cfg: NodeEncoderConfig):
        super().__init__()
        self.cfg = cfg
        T = cfg.n_patches
        d_patch = cfg.d_input // T
        if cfg.d_input % T != 0:
            raise ValueError(f"d_input={cfg.d_input} must be divisible by n_patches={T}")

        self.patch_proj = nn.Linear(d_patch, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, T + 1, cfg.d_model))

        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,  # pre-norm when supported
            )
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        x: (B, d_input)
        return_sequence=False: returns h: (B, d_model)  — CLS token output
        return_sequence=True:  returns   (B, T+1, d_model) — full token sequence
        """
        B = x.shape[0]
        T = self.cfg.n_patches
        d_patch = self.cfg.d_input // T

        # Tokenize: (B, T, d_patch) -> (B, T, d_model)
        patches = x.reshape(B, T, d_patch)
        tokens = self.patch_proj(patches)

        # Prepend CLS: (B, T+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Transformer: (B, T+1, d_model)
        out = self.transformer(tokens)

        if return_sequence:
            return out  # (B, T+1, d_model)
        return out[:, 0, :]  # (B, d_model)
