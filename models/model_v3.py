"""Full MultimodalContrastiveModel: 3 NodeEncoders + ProjectionHeads + optional FusionMLPs + Classifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import NodeEncoder, NodeEncoderConfig
from models.projection import ProjectionHead
from models.fusion_mlp import FusionMLP
from models.decoder import RawDecoder, EmbPredictor
from models.cross_modal_transformer import CrossModalTransformer, CrossModalTransformerConfig


@dataclass
class ModelV3Config:
    # Encoder
    d_input: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    n_patches: int = 4
    encoder_dropout: float = 0.0
    share_encoder_weights: bool = False

    # Projection head
    d_z: int = 64
    proj_hidden: int = 128
    fused_hidden: int = 128

    # Fusion MLP (ConFu)
    fusion_dropout: float = 0.1
    fusion_mask_prob: float = 0.0

    # Masked prediction decoders
    mask_ratio: float = 0.5
    decoder_hidden: int = 128

    # Cross-modal Transformer (for masked_raw / masked_emb)
    use_cross_modal_tf: bool = False
    cross_modal_n_layers: int = 2
    cross_modal_n_heads: int = 4
    cross_modal_d_ff: int = 128
    cross_modal_dropout: float = 0.0

    # Classification head
    classifier_hidden: int = 0
    n_classes: int = 7   # Q^n_active_atoms


class MultimodalContrastiveModel(nn.Module):
    """
    Three independent NodeEncoders with ProjectionHeads and an optional classification head.
    Supports ConFu-style FusionMLPs for higher-order contrastive terms.
    """

    def __init__(self, cfg: ModelV3Config):
        super().__init__()
        self.cfg = cfg

        enc_cfg = NodeEncoderConfig(
            d_input=cfg.d_input,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            n_patches=cfg.n_patches,
            dropout=cfg.encoder_dropout,
        )

        if cfg.share_encoder_weights:
            shared_enc = NodeEncoder(enc_cfg)
            self.encoders = nn.ModuleList([shared_enc, shared_enc, shared_enc])
        else:
            self.encoders = nn.ModuleList([NodeEncoder(enc_cfg) for _ in range(3)])

        self.proj_heads = nn.ModuleList([
            ProjectionHead(cfg.d_model, cfg.proj_hidden, cfg.d_z) for _ in range(3)
        ])

        self.full_fusion = nn.Sequential(
            nn.Linear(3 * cfg.d_z, cfg.fused_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fused_hidden, cfg.d_z),
        )

        # ConFu fusion MLPs: one per pair (12, 13, 23)
        self.fusion_mlps = nn.ModuleDict({
            "f12": FusionMLP(cfg.d_z, cfg.fusion_dropout),
            "f13": FusionMLP(cfg.d_z, cfg.fusion_dropout),
            "f23": FusionMLP(cfg.d_z, cfg.fusion_dropout),
        })

        # Decoders for masked_raw
        self.raw_dec1 = RawDecoder(cfg.d_z, cfg.decoder_hidden, cfg.d_input)
        self.raw_dec2 = RawDecoder(cfg.d_z, cfg.decoder_hidden, cfg.d_input)
        self.raw_dec3 = RawDecoder(cfg.d_z, cfg.decoder_hidden, cfg.d_input)

        # Predictors for masked_emb
        self.emb_pred1 = EmbPredictor(cfg.d_model, cfg.decoder_hidden, cfg.d_model)
        self.emb_pred2 = EmbPredictor(cfg.d_model, cfg.decoder_hidden, cfg.d_model)
        self.emb_pred3 = EmbPredictor(cfg.d_model, cfg.decoder_hidden, cfg.d_model)

        # Optional cross-modal Transformer for masked prediction
        if cfg.use_cross_modal_tf:
            cm_cfg = CrossModalTransformerConfig(
                d_model=cfg.d_model,
                n_heads=cfg.cross_modal_n_heads,
                n_layers=cfg.cross_modal_n_layers,
                d_ff=cfg.cross_modal_d_ff,
                n_modalities=3,
                dropout=cfg.cross_modal_dropout,
            )
            self.cross_modal_tf = CrossModalTransformer(cm_cfg)
        else:
            self.cross_modal_tf = None

        # Classification head: optionally nonlinear for harder supervised targets.
        if cfg.classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(3 * cfg.d_z, cfg.classifier_hidden),
                nn.ReLU(),
                nn.Linear(cfg.classifier_hidden, cfg.n_classes),
            )
        else:
            self.classifier = nn.Linear(3 * cfg.d_z, cfg.n_classes)

    def encode(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns encoder representations (h1, h2, h3), each (B, d_model)."""
        h1 = self.encoders[0](x1)
        h2 = self.encoders[1](x2)
        h3 = self.encoders[2](x3)
        return h1, h2, h3

    def cross_modal_encode(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode inputs through NodeEncoders (returning full sequences), then
        pass concatenated sequences through the CrossModalTransformer.
        Returns cross-modal CLS representations (cm_h1, cm_h2, cm_h3), each (B, d_model).
        Requires use_cross_modal_tf=True.
        """
        seq1 = self.encoders[0](x1, return_sequence=True)  # (B, T+1, d_model)
        seq2 = self.encoders[1](x2, return_sequence=True)
        seq3 = self.encoders[2](x3, return_sequence=True)
        return self.cross_modal_tf([seq1, seq2, seq3])      # (B, d_model) each

    def project(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns L2-normalized projections (z1, z2, z3), each (B, d_z)."""
        z1 = self.proj_heads[0](h1)
        z2 = self.proj_heads[1](h2)
        z3 = self.proj_heads[2](h3)
        return z1, z2, z3

    def fuse(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns fused representations for each pair."""
        mask_prob = self.cfg.fusion_mask_prob if self.training else 0.0
        return {
            "f12": self.fusion_mlps["f12"](z1, z2, mask_prob),
            "f13": self.fusion_mlps["f13"](z1, z3, mask_prob),
            "f23": self.fusion_mlps["f23"](z2, z3, mask_prob),
        }

    def fuse_all(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        """Returns one normalized fused embedding for the full 3-view input."""
        fused = self.full_fusion(torch.cat([z1, z2, z3], dim=-1))
        return F.normalize(fused, dim=-1)

    def classify(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        """Classification logits from concatenated projections."""
        return self.classifier(torch.cat([z1, z2, z3], dim=-1))

    def forward(
        self,
        x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
        x1_full: Optional[torch.Tensor] = None,
        x2_full: Optional[torch.Tensor] = None,
        x3_full: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        Returns dict with h1/h2/h3 (encoder), z1/z2/z3 (projections), logits.

        When x{1,2,3}_full are provided (masked methods), also encodes the full
        (unmasked) inputs and returns z{1,2,3}_full plus decoder/predictor modules.
        """
        h1, h2, h3 = self.encode(x1, x2, x3)
        z1, z2, z3 = self.project(h1, h2, h3)
        logits = self.classify(z1, z2, z3)
        out: Dict[str, torch.Tensor] = {
            "h1": h1, "h2": h2, "h3": h3,
            "z1": z1, "z2": z2, "z3": z3,
            "logits": logits,
        }

        if x1_full is not None:
            # Encode full (unmasked) inputs for teacher
            h1f, h2f, h3f = self.encode(x1_full, x2_full, x3_full)
            z1f, z2f, z3f = self.project(h1f, h2f, h3f)
            out["h1_masked"] = h1
            out["h2_masked"] = h2
            out["h3_masked"] = h3
            out["z1_masked"] = z1
            out["z2_masked"] = z2
            out["z3_masked"] = z3
            out["h1_full"] = h1f
            out["h2_full"] = h2f
            out["h3_full"] = h3f
            out["z1_full"] = z1f
            out["z2_full"] = z2f
            out["z3_full"] = z3f
            out["raw_decs"] = (self.raw_dec1, self.raw_dec2, self.raw_dec3)
            out["emb_preds"] = (self.emb_pred1, self.emb_pred2, self.emb_pred3)

            # Cross-modal transformer: encode masked and full inputs jointly
            if self.cross_modal_tf is not None:
                cm_h1, cm_h2, cm_h3 = self.cross_modal_encode(x1, x2, x3)
                cm_h1f, cm_h2f, cm_h3f = self.cross_modal_encode(x1_full, x2_full, x3_full)
                out["cm_h1_masked"] = cm_h1    # cross-attended, from masked inputs
                out["cm_h2_masked"] = cm_h2
                out["cm_h3_masked"] = cm_h3
                out["cm_h1_full"] = cm_h1f     # cross-attended, from full inputs (teacher)
                out["cm_h2_full"] = cm_h2f
                out["cm_h3_full"] = cm_h3f

        return out
