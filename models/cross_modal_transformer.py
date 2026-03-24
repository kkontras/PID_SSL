"""CrossModalTransformer: joint Transformer over concatenated per-modality token sequences."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
)


@dataclass
class CrossModalTransformerConfig:
    d_model: int = 64       # must match NodeEncoder d_model
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    n_modalities: int = 3
    dropout: float = 0.0


class CrossModalTransformer(nn.Module):
    """
    Joint Transformer over concatenated per-modality token sequences.

    After each NodeEncoder produces a sequence (B, T+1, d_model), this module:
      1. Adds a learnable modality embedding to every token in each sequence
         so the joint Transformer can distinguish which modality each token belongs to.
      2. Concatenates all sequences along the token dimension:
         (B, n_modalities*(T+1), d_model)
      3. Runs N Transformer encoder layers (full cross-modal attention).
      4. Splits the output back into per-modality sequences and returns the
         CLS token (position 0) of each as the new cross-modal representation.

    The CLS outputs (B, d_model) are a drop-in replacement for the individual
    NodeEncoder CLS tokens and are used by the masked prediction losses.
    """

    def __init__(self, cfg: CrossModalTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # One modality embedding per modality, broadcast over batch and sequence
        # shape: (n_modalities, 1, d_model) — added to (B, T+1, d_model)
        self.modality_embed = nn.Parameter(
            torch.zeros(cfg.n_modalities, 1, cfg.d_model)
        )

        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
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
        nn.init.trunc_normal_(self.modality_embed, std=0.02)

    def forward(
        self, sequences: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sequences: list of n_modalities tensors, each (B, T+1, d_model)

        Returns:
            h1, h2, h3: each (B, d_model) — CLS token output per modality
                        after cross-modal attention.
        """
        assert len(sequences) == self.cfg.n_modalities

        # Add per-modality embedding to every token in that modality's sequence
        # modality_embed[i]: (1, d_model) -> broadcast to (B, T+1, d_model)
        tagged = [
            seq + self.modality_embed[i].unsqueeze(0)  # (1, 1, d_model) broadcast
            for i, seq in enumerate(sequences)
        ]

        # Concatenate along sequence dim: (B, n_modalities*(T+1), d_model)
        tokens = torch.cat(tagged, dim=1)

        # Joint cross-modal Transformer
        out = self.transformer(tokens)  # (B, n_modalities*(T+1), d_model)

        # Split back into per-modality sequences
        seq_len = sequences[0].shape[1]  # T+1 (same for all modalities)
        splits = out.split(seq_len, dim=1)  # tuple of (B, T+1, d_model)

        # Return CLS token (position 0) of each modality
        return splits[0][:, 0, :], splits[1][:, 0, :], splits[2][:, 0, :]
