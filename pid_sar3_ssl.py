from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSLEncoderConfig:
    input_dim: int = 32
    encoder_hidden_dim: int = 128
    representation_dim: int = 64
    projector_hidden_dim: int = 128
    projector_dim: int = 64


@dataclass
class SSLTrainConfig:
    objective: str = "pairwise_simclr"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    steps: int = 200
    temperature: float = 0.2
    log_every: int = 10
    device: str = "cpu"
    seed: int = 0
    triangle_reg_weight: float = 0.15
    confu_pair_weight: float = 0.5
    confu_fused_weight: float = 0.5
    directional_pred_weight: float = 0.5


@dataclass
class VectorAugmentationConfig:
    jitter_std: float = 0.08
    feature_drop_prob: float = 0.10
    gain_min: float = 0.9
    gain_max: float = 1.1


class VectorAugmenter(nn.Module):
    """Simple SimCLR-style augmentations for synthetic vector observations."""

    def __init__(self, cfg: Optional[VectorAugmentationConfig] = None):
        super().__init__()
        self.cfg = cfg or VectorAugmentationConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        out = x
        if self.cfg.feature_drop_prob > 0.0:
            keep = (torch.rand_like(out) > self.cfg.feature_drop_prob).float()
            out = out * keep
        if self.cfg.gain_max > self.cfg.gain_min:
            gains = torch.empty((out.shape[0], 1), device=out.device).uniform_(self.cfg.gain_min, self.cfg.gain_max)
            out = out * gains
        if self.cfg.jitter_std > 0.0:
            out = out + self.cfg.jitter_std * torch.randn_like(out)
        return out


class ModalityEncoder(nn.Module):
    def __init__(self, cfg: SSLEncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder_hidden_dim, cfg.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder_hidden_dim, cfg.representation_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Projector(nn.Module):
    def __init__(self, cfg: SSLEncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.representation_dim, cfg.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.projector_hidden_dim, cfg.projector_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class FusionHead(nn.Module):
    """Trainable fusion head for pair -> target-style fused contrastive terms."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b], dim=-1))


class TriModalSSLModel(nn.Module):
    """Three independent encoders/projectors for x1, x2, x3 modalities."""

    def __init__(self, cfg: SSLEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict(
            {
                "x1": ModalityEncoder(cfg),
                "x2": ModalityEncoder(cfg),
                "x3": ModalityEncoder(cfg),
            }
        )
        self.projectors = nn.ModuleDict(
            {
                "x1": Projector(cfg),
                "x2": Projector(cfg),
                "x3": Projector(cfg),
            }
        )
        # Pair-fusion heads for ConFu-style fused higher-order alignment.
        self.fusion_heads = nn.ModuleDict(
            {
                "x23_to_x1": FusionHead(cfg.projector_dim, cfg.projector_hidden_dim),
                "x13_to_x2": FusionHead(cfg.projector_dim, cfg.projector_hidden_dim),
                "x12_to_x3": FusionHead(cfg.projector_dim, cfg.projector_hidden_dim),
            }
        )
        # Directional predictive heads on encoder representation space h (not projector z).
        self.directional_heads = nn.ModuleDict(
            {
                "h23_to_h1": FusionHead(cfg.representation_dim, cfg.encoder_hidden_dim),
                "h13_to_h2": FusionHead(cfg.representation_dim, cfg.encoder_hidden_dim),
                "h12_to_h3": FusionHead(cfg.representation_dim, cfg.encoder_hidden_dim),
            }
        )

    def encode(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.encoders[k](batch[k]) for k in ("x1", "x2", "x3")}

    def project(self, reps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.projectors[k](reps[k]) for k in ("x1", "x2", "x3")}

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        h = self.encode(batch)
        z = self.project(h)
        return h, z

    def fuse_projected(self, z: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "to_x1": self.fusion_heads["x23_to_x1"](z["x2"], z["x3"]),
            "to_x2": self.fusion_heads["x13_to_x2"](z["x1"], z["x3"]),
            "to_x3": self.fusion_heads["x12_to_x3"](z["x1"], z["x2"]),
        }

    def predict_directional_h(self, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "to_h1": self.directional_heads["h23_to_h1"](h["x2"], h["x3"]),
            "to_h2": self.directional_heads["h13_to_h2"](h["x1"], h["x3"]),
            "to_h3": self.directional_heads["h12_to_h3"](h["x1"], h["x2"]),
        }


class UnimodalSimCLRModel(nn.Module):
    def __init__(self, cfg: SSLEncoderConfig):
        super().__init__()
        self.encoder = ModalityEncoder(cfg)
        self.projector = Projector(cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def _nt_xent_pair(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Standard SimCLR / NT-Xent loss for a positive paired batch (a_i, b_i)."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    n = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)  # (2n, d)
    logits = (z @ z.T) / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()

    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(2 * n, device=z.device)
    targets[:n] += n
    targets[n:] -= n
    return F.cross_entropy(logits, targets)


def _multi_positive_infonce(anchor: torch.Tensor, positives_1: torch.Tensor, positives_2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Anchor has two positives for the same index (e.g., x1 matches x2 and x3).
    Negatives are all non-matching examples from the two positive pools.
    """
    anchor = F.normalize(anchor, dim=-1)
    positives_1 = F.normalize(positives_1, dim=-1)
    positives_2 = F.normalize(positives_2, dim=-1)
    n = anchor.shape[0]

    cand = torch.cat([positives_1, positives_2], dim=0)  # (2n, d)
    logits = (anchor @ cand.T) / temperature  # (n, 2n)
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1) + 1e-12
    pos_mass = exp_logits[torch.arange(n, device=anchor.device), torch.arange(n, device=anchor.device)]
    pos_mass = pos_mass + exp_logits[torch.arange(n, device=anchor.device), torch.arange(n, device=anchor.device) + n]
    return (-torch.log((pos_mass + 1e-12) / denom)).mean()


def _triangle_shape_score(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
    """
    Dimensionless triangle shape score in [0, 1] (approx; numerical clipping used),
    higher means less collinear and more equilateral.
    """
    p1 = F.normalize(z1, dim=-1)
    p2 = F.normalize(z2, dim=-1)
    p3 = F.normalize(z3, dim=-1)
    a = torch.linalg.norm(p2 - p3, dim=-1)
    b = torch.linalg.norm(p1 - p3, dim=-1)
    c = torch.linalg.norm(p1 - p2, dim=-1)
    s = 0.5 * (a + b + c)
    area_sq = torch.clamp(s * (s - a) * (s - b) * (s - c), min=0.0)
    area = torch.sqrt(area_sq + 1e-12)
    # Normalized shape factor (1 for equilateral triangle).
    denom = (a * a + b * b + c * c) + 1e-12
    score = (4.0 * np.sqrt(3.0) * area) / denom
    return torch.clamp(score, 0.0, 1.0)


def _fused_pair(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    return F.normalize(0.5 * (z_a + z_b), dim=-1)


def _triangle_area(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Triangle area A(x,y,z) with unit-normalized embeddings (paper-aligned similarity core)."""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    z = F.normalize(z, dim=-1)
    u = x - y
    v = x - z
    uu = torch.sum(u * u, dim=-1)
    vv = torch.sum(v * v, dim=-1)
    uv = torch.sum(u * v, dim=-1)
    area_sq = torch.clamp(uu * vv - uv * uv, min=0.0)
    return 0.5 * torch.sqrt(area_sq + 1e-12)


def _triangle_logits_vary_target(target: torch.Tensor, p: torch.Tensor, q: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Logits[i,j] = -A(target_j, p_i, q_i)/tau
    Paper analogue: vary one modality while keeping the paired modalities fixed.
    """
    t = F.normalize(target, dim=-1)
    p = F.normalize(p, dim=-1)
    q = F.normalize(q, dim=-1)
    tj = t.unsqueeze(0)  # (1,B,D)
    pi = p.unsqueeze(1)  # (B,1,D)
    qi = q.unsqueeze(1)  # (B,1,D)
    u = tj - pi
    v = tj - qi
    uu = torch.sum(u * u, dim=-1)
    vv = torch.sum(v * v, dim=-1)
    uv = torch.sum(u * v, dim=-1)
    area = 0.5 * torch.sqrt(torch.clamp(uu * vv - uv * uv, min=0.0) + 1e-12)
    return -area / temperature


def _triangle_cross_entropy(logits: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def _triangle_exact_symmetric_loss(z: Dict[str, torch.Tensor], temperature: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Symmetric 3-modal TRIANGLE-style contrastive loss inspired by paper Eqs. (5)-(6):
    for each target modality k with paired modalities (i,j), include:
    - vary-target term:   A(target_j, pair_i_fixed)
    - vary-pair term:     A(target_i_fixed, pair_j)
    """
    keys = [("x1", "x2", "x3"), ("x2", "x1", "x3"), ("x3", "x1", "x2")]
    terms = {}
    losses = []
    for tgt, p, q in keys:
        logits_vary_t = _triangle_logits_vary_target(z[tgt], z[p], z[q], temperature)
        lv = _triangle_cross_entropy(logits_vary_t)
        # vary pair jointly while target fixed = transpose candidate structure
        logits_vary_pair = _triangle_logits_vary_target(z[tgt], z[p], z[q], temperature).T
        lp = _triangle_cross_entropy(logits_vary_pair)
        terms[f"tri_{p}{q}_to_{tgt}"] = float(lv.detach().cpu())
        terms[f"tri_{tgt}_to_{p}{q}"] = float(lp.detach().cpu())
        losses.extend([lv, lp])
    loss = sum(losses) / float(len(losses))
    # Mean positive triangle area as a useful diagnostic (lower is tighter triplets).
    pos_area = _triangle_area(z["x1"], z["x2"], z["x3"]).mean()
    terms["triangle_pos_area"] = float(pos_area.detach().cpu())
    return loss, terms


def ssl_objective_loss(
    z: Dict[str, torch.Tensor],
    objective: str,
    temperature: float = 0.2,
    triangle_reg_weight: float = 0.15,
    confu_pair_weight: float = 0.5,
    confu_fused_weight: float = 0.5,
    fused: Optional[Dict[str, torch.Tensor]] = None,
    h: Optional[Dict[str, torch.Tensor]] = None,
    directional_preds: Optional[Dict[str, torch.Tensor]] = None,
    directional_pred_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if objective == "pairwise_simclr":
        l12 = _nt_xent_pair(z["x1"], z["x2"], temperature)
        l13 = _nt_xent_pair(z["x1"], z["x3"], temperature)
        l23 = _nt_xent_pair(z["x2"], z["x3"], temperature)
        loss = (l12 + l13 + l23) / 3.0
        metrics = {"loss_12": float(l12.detach().cpu()), "loss_13": float(l13.detach().cpu()), "loss_23": float(l23.detach().cpu())}
        return loss, metrics
    if objective == "tri_positive_infonce":
        l1 = _multi_positive_infonce(z["x1"], z["x2"], z["x3"], temperature)
        l2 = _multi_positive_infonce(z["x2"], z["x1"], z["x3"], temperature)
        l3 = _multi_positive_infonce(z["x3"], z["x1"], z["x2"], temperature)
        loss = (l1 + l2 + l3) / 3.0
        metrics = {"loss_anchor_x1": float(l1.detach().cpu()), "loss_anchor_x2": float(l2.detach().cpu()), "loss_anchor_x3": float(l3.detach().cpu())}
        return loss, metrics
    if objective == "triangle_like":
        l12 = _nt_xent_pair(z["x1"], z["x2"], temperature)
        l13 = _nt_xent_pair(z["x1"], z["x3"], temperature)
        l23 = _nt_xent_pair(z["x2"], z["x3"], temperature)
        pair_loss = (l12 + l13 + l23) / 3.0
        shape = _triangle_shape_score(z["x1"], z["x2"], z["x3"]).mean()
        # Encourage non-collinear tri-modal geometry while retaining contrastive pair matching.
        loss = pair_loss - triangle_reg_weight * shape
        metrics = {
            "pair_loss": float(pair_loss.detach().cpu()),
            "triangle_shape": float(shape.detach().cpu()),
            "loss_12": float(l12.detach().cpu()),
            "loss_13": float(l13.detach().cpu()),
            "loss_23": float(l23.detach().cpu()),
        }
        return loss, metrics
    if objective == "triangle_exact":
        return _triangle_exact_symmetric_loss(z, temperature)
    if objective == "confu_style":
        l12 = _nt_xent_pair(z["x1"], z["x2"], temperature)
        l13 = _nt_xent_pair(z["x1"], z["x3"], temperature)
        l23 = _nt_xent_pair(z["x2"], z["x3"], temperature)
        pair_loss = (l12 + l13 + l23) / 3.0

        if fused is None:
            # Backward-compatible fallback (non-trainable fusion) if no fusion heads are provided.
            f23 = _fused_pair(z["x2"], z["x3"])
            f13 = _fused_pair(z["x1"], z["x3"])
            f12 = _fused_pair(z["x1"], z["x2"])
            lf1 = _nt_xent_pair(f23, z["x1"], temperature)
            lf2 = _nt_xent_pair(f13, z["x2"], temperature)
            lf3 = _nt_xent_pair(f12, z["x3"], temperature)
        else:
            lf1 = _nt_xent_pair(fused["to_x1"], z["x1"], temperature)
            lf2 = _nt_xent_pair(fused["to_x2"], z["x2"], temperature)
            lf3 = _nt_xent_pair(fused["to_x3"], z["x3"], temperature)
        fused_loss = (lf1 + lf2 + lf3) / 3.0

        loss = confu_pair_weight * pair_loss + confu_fused_weight * fused_loss
        metrics = {
            "pair_loss": float(pair_loss.detach().cpu()),
            "fused_loss": float(fused_loss.detach().cpu()),
            "fuse_to_x1": float(lf1.detach().cpu()),
            "fuse_to_x2": float(lf2.detach().cpu()),
            "fuse_to_x3": float(lf3.detach().cpu()),
        }
        return loss, metrics
    if objective == "directional_predictive_hybrid":
        # Pairwise cross-modal contrastive baseline term on projector space.
        l12 = _nt_xent_pair(z["x1"], z["x2"], temperature)
        l13 = _nt_xent_pair(z["x1"], z["x3"], temperature)
        l23 = _nt_xent_pair(z["x2"], z["x3"], temperature)
        pair_loss = (l12 + l13 + l23) / 3.0

        if h is None or directional_preds is None:
            raise ValueError("directional_predictive_hybrid requires h and directional_preds")

        # Directional prediction on h-space (targets stop-gradient by construction through detach()).
        def _pred_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            pred_n = F.normalize(pred, dim=-1)
            tgt_n = F.normalize(target.detach(), dim=-1)
            return (1.0 - torch.sum(pred_n * tgt_n, dim=-1)).mean()

        p1 = _pred_loss(directional_preds["to_h1"], h["x1"])
        p2 = _pred_loss(directional_preds["to_h2"], h["x2"])
        p3 = _pred_loss(directional_preds["to_h3"], h["x3"])
        pred_loss = (p1 + p2 + p3) / 3.0
        loss = pair_loss + directional_pred_weight * pred_loss
        metrics = {
            "pair_loss": float(pair_loss.detach().cpu()),
            "directional_pred_loss": float(pred_loss.detach().cpu()),
            "pred_23_to_1": float(p1.detach().cpu()),
            "pred_13_to_2": float(p2.detach().cpu()),
            "pred_12_to_3": float(p3.detach().cpu()),
        }
        return loss, metrics
    raise ValueError(f"Unknown objective: {objective}")


def numpy_batch_to_torch(batch: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in ("x1", "x2", "x3"):
        out[k] = torch.from_numpy(batch[k]).float().to(device)
    return out


def train_ssl(
    model: TriModalSSLModel,
    generator,
    cfg: SSLTrainConfig,
    pid_schedule: Optional[Iterable[int]] = None,
) -> List[Dict[str, float]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    schedule_list = list(pid_schedule) if pid_schedule is not None else None
    history: List[Dict[str, float]] = []
    for step in range(1, cfg.steps + 1):
        if schedule_list is None:
            batch_np = generator.generate(n=cfg.batch_size)
        else:
            pids = [schedule_list[(step * cfg.batch_size + i) % len(schedule_list)] for i in range(cfg.batch_size)]
            batch_np = generator.generate(n=cfg.batch_size, pid_ids=pids)
        batch_t = numpy_batch_to_torch(batch_np, device=str(device))
        _, z = model(batch_t)
        fused = None
        directional_preds = None
        if cfg.objective == "confu_style" and hasattr(model, "fuse_projected"):
            fused = model.fuse_projected(z)  # type: ignore[attr-defined]
        if cfg.objective == "directional_predictive_hybrid" and hasattr(model, "predict_directional_h"):
            h_for_pred = model.encode(batch_t)  # recompute h for clarity in objective branch
            directional_preds = model.predict_directional_h(h_for_pred)  # type: ignore[attr-defined]
        else:
            h_for_pred = None
        loss, parts = ssl_objective_loss(
            z,
            objective=cfg.objective,
            temperature=cfg.temperature,
            triangle_reg_weight=cfg.triangle_reg_weight,
            confu_pair_weight=cfg.confu_pair_weight,
            confu_fused_weight=cfg.confu_fused_weight,
            fused=fused,
            h=h_for_pred,
            directional_preds=directional_preds,
            directional_pred_weight=cfg.directional_pred_weight,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        row = {"step": float(step), "loss": float(loss.detach().cpu())}
        row.update(parts)
        history.append(row)
    return history


def train_unimodal_simclr(
    model: UnimodalSimCLRModel,
    generator,
    modality_key: str,
    cfg: SSLTrainConfig,
    augmenter: Optional[VectorAugmenter] = None,
    pid_schedule: Optional[Iterable[int]] = None,
) -> List[Dict[str, float]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    model.train()
    aug = augmenter or VectorAugmenter()
    aug.to(device)
    aug.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    schedule_list = list(pid_schedule) if pid_schedule is not None else None
    history: List[Dict[str, float]] = []
    for step in range(1, cfg.steps + 1):
        if schedule_list is None:
            batch_np = generator.generate(n=cfg.batch_size)
        else:
            pids = [schedule_list[(step * cfg.batch_size + i) % len(schedule_list)] for i in range(cfg.batch_size)]
            batch_np = generator.generate(n=cfg.batch_size, pid_ids=pids)

        x = torch.from_numpy(batch_np[modality_key]).float().to(device)
        x1 = aug(x)
        x2 = aug(x)
        _, z1 = model(x1)
        _, z2 = model(x2)
        loss = _nt_xent_pair(z1, z2, temperature=cfg.temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        history.append({"step": float(step), "loss": float(loss.detach().cpu())})
    return history


@torch.no_grad()
def encode_numpy(model: TriModalSSLModel, batch: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, np.ndarray]:
    model.eval()
    t_batch = numpy_batch_to_torch(batch, device=device)
    reps = model.encode(t_batch)
    return {k: reps[k].detach().cpu().numpy().astype(np.float32) for k in ("x1", "x2", "x3")}


@torch.no_grad()
def encode_unimodal_numpy(
    model: UnimodalSimCLRModel,
    x: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    xt = torch.from_numpy(x).float().to(device)
    h = model.encoder(xt)
    return h.detach().cpu().numpy().astype(np.float32)


def concat_representations(reps: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([reps["x1"], reps["x2"], reps["x3"]], axis=1)


def family_from_pid_ids(pid_ids: np.ndarray) -> np.ndarray:
    pid_ids = np.asarray(pid_ids).astype(np.int64)
    fam = np.zeros_like(pid_ids)
    fam[(pid_ids >= 3) & (pid_ids <= 6)] = 1  # redundancy
    fam[(pid_ids >= 7)] = 2  # synergy
    return fam


def pair_retrieval_at_1(a: np.ndarray, b: np.ndarray) -> float:
    """Same-index retrieval accuracy using cosine similarity."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sims = a_n @ b_n.T
    pred = np.argmax(sims, axis=1)
    target = np.arange(a.shape[0])
    return float(np.mean(pred == target))


def pair_retrieval_matrix(reps: Dict[str, np.ndarray]) -> np.ndarray:
    keys = ("x1", "x2", "x3")
    mat = np.zeros((3, 3), dtype=np.float32)
    for i, ka in enumerate(keys):
        for j, kb in enumerate(keys):
            if i == j:
                mat[i, j] = 1.0
            else:
                mat[i, j] = pair_retrieval_at_1(reps[ka], reps[kb])
    return mat
