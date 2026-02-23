from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PID_ID_TO_NAME = {
    0: "U1",
    1: "U2",
    2: "U3",
    3: "R12",
    4: "R13",
    5: "R23",
    6: "R123",
    7: "S12->3",
    8: "S13->2",
    9: "S23->1",
}


@dataclass
class PIDDatasetConfig:
    d: int = 32
    m: int = 8
    sigma: float = 0.5
    alpha_min: float = 0.8
    alpha_max: float = 1.2
    rho_choices: Tuple[float, ...] = (0.2, 0.5, 0.8)
    hop_choices: Tuple[int, ...] = (1, 2, 3, 4)
    max_hop: int = 4
    seed: int = 0
    deleakage_fit_samples: int = 2048
    deleakage_ridge: float = 1e-4
    synergy_hidden_scale: float = 1.0
    unique_gain: float = 1.0
    redundancy_gain: float = 1.0
    synergy_gain: float = 1.0
    pid_gain_overrides: Optional[Dict[int, float]] = None


class FixedSynergyMLP:
    """Fixed random residual MLP used to synthesize directional synergy."""

    def __init__(self, m: int, hmax: int, rng: np.random.Generator, hidden_scale: float = 1.0):
        self.m = m
        self.hmax = hmax
        self.hidden = 2 * m
        self.in_w = rng.normal(0.0, hidden_scale / np.sqrt(2 * m), size=(2 * m, self.hidden))
        self.in_b = rng.normal(0.0, 0.05, size=(self.hidden,))

        self.blocks = []
        for _ in range(hmax):
            block = {
                "w1": rng.normal(0.0, hidden_scale / np.sqrt(self.hidden), size=(self.hidden, self.hidden)),
                "b1": rng.normal(0.0, 0.05, size=(self.hidden,)),
                "w2": rng.normal(0.0, hidden_scale / np.sqrt(self.hidden), size=(self.hidden, self.hidden)),
                "b2": rng.normal(0.0, 0.05, size=(self.hidden,)),
            }
            self.blocks.append(block)

        self.out_w = rng.normal(0.0, hidden_scale / np.sqrt(self.hidden), size=(self.hidden, m))
        self.out_b = rng.normal(0.0, 0.05, size=(m,))

    @staticmethod
    def _act(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def __call__(self, a: np.ndarray, b: np.ndarray, hop: int) -> np.ndarray:
        x = np.concatenate([a, b], axis=-1)
        h = self._act(x @ self.in_w + self.in_b)
        for idx in range(hop):
            blk = self.blocks[idx]
            z = self._act(h @ blk["w1"] + blk["b1"])
            z = self._act(z @ blk["w2"] + blk["b2"])
            h = self._act(h + z)
        out = self._act(h @ self.out_w + self.out_b)
        return out


class PIDSar3DatasetGenerator:
    """
    Synthetic 3-view generator where each sample contains one PID atom.

    Returns raw observations x1, x2, x3 and metadata matching the plan.
    """

    COMPONENTS = (
        "U1",
        "U2",
        "U3",
        "R12",
        "R13",
        "R23",
        "R123",
        "A12",
        "B12",
        "A13",
        "B13",
        "A23",
        "B23",
        "SYN12",
        "SYN13",
        "SYN23",
    )

    def __init__(self, config: Optional[PIDDatasetConfig] = None):
        self.cfg = config or PIDDatasetConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.P = self._sample_projections()
        self.synergy_mlp = FixedSynergyMLP(
            m=self.cfg.m,
            hmax=self.cfg.max_hop,
            rng=self.rng,
            hidden_scale=self.cfg.synergy_hidden_scale,
        )
        self.C_a, self.C_b = self._fit_deleakage_maps()

    def _sample_projections(self) -> Dict[int, Dict[str, np.ndarray]]:
        d, m = self.cfg.d, self.cfg.m
        proj: Dict[int, Dict[str, np.ndarray]] = {1: {}, 2: {}, 3: {}}
        for view in (1, 2, 3):
            for comp in self.COMPONENTS:
                mat = self.rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, m))
                col_norms = np.linalg.norm(mat, axis=0, keepdims=True) + 1e-8
                proj[view][comp] = mat / col_norms
        return proj

    def _fit_deleakage_maps(self) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        m = self.cfg.m
        n = self.cfg.deleakage_fit_samples
        ridge = self.cfg.deleakage_ridge
        C_a: Dict[int, np.ndarray] = {}
        C_b: Dict[int, np.ndarray] = {}

        for hop in range(1, self.cfg.max_hop + 1):
            a = self.rng.normal(size=(n, m))
            b = self.rng.normal(size=(n, m))
            s0 = np.stack([self.synergy_mlp(ai, bi, hop) for ai, bi in zip(a, b)], axis=0)
            X = np.concatenate([a, b], axis=1)  # (n, 2m)
            XtX = X.T @ X + ridge * np.eye(2 * m)
            W = np.linalg.solve(XtX, X.T @ s0)  # (2m, m)
            C_a[hop] = W[:m, :].copy()
            C_b[hop] = W[m:, :].copy()
        return C_a, C_b

    def _noise(self) -> np.ndarray:
        return self.rng.normal(0.0, self.cfg.sigma, size=(self.cfg.d,))

    def _sample_alpha(self) -> float:
        return float(self.rng.uniform(self.cfg.alpha_min, self.cfg.alpha_max))

    def _sample_rho(self) -> float:
        return float(self.rng.choice(np.asarray(self.cfg.rho_choices)))

    def _sample_hop(self) -> int:
        return int(self.rng.choice(np.asarray(self.cfg.hop_choices)))

    def _atom_gain(self, pid_id: int) -> float:
        if self.cfg.pid_gain_overrides is not None and pid_id in self.cfg.pid_gain_overrides:
            return float(self.cfg.pid_gain_overrides[pid_id])
        if pid_id <= 2:
            return float(self.cfg.unique_gain)
        if pid_id <= 6:
            return float(self.cfg.redundancy_gain)
        return float(self.cfg.synergy_gain)

    def _project(self, view: int, comp: str, latent: np.ndarray, alpha: float) -> np.ndarray:
        return alpha * (self.P[view][comp] @ latent)

    def sample(self, pid_id: Optional[int] = None, return_aux: bool = False) -> Dict[str, np.ndarray]:
        if pid_id is None:
            pid_id = int(self.rng.integers(0, 10))

        if pid_id not in PID_ID_TO_NAME:
            raise ValueError(f"Invalid pid_id={pid_id}")

        alpha = self._sample_alpha() * self._atom_gain(pid_id)
        sigma = float(self.cfg.sigma)
        rho = -1.0
        hop = 0

        x1 = np.zeros(self.cfg.d, dtype=np.float32)
        x2 = np.zeros(self.cfg.d, dtype=np.float32)
        x3 = np.zeros(self.cfg.d, dtype=np.float32)
        aux: Dict[str, np.ndarray] = {}
        if return_aux:
            aux = {
                "y_u1": np.float32(0.0),
                "mask_y_u1": np.int64(0),
                "y_r12": np.float32(0.0),
                "mask_y_r12": np.int64(0),
                "y_r123": np.float32(0.0),
                "mask_y_r123": np.int64(0),
                "y_s12_3": np.float32(0.0),
                "mask_y_s12_3": np.int64(0),
            }

        name = PID_ID_TO_NAME[pid_id]

        if name in ("U1", "U2", "U3"):
            u = self.rng.normal(size=(self.cfg.m,))
            target_view = int(name[-1])
            x = self._project(target_view, name, u, alpha)
            if return_aux and pid_id == 0:
                aux["y_u1"] = np.float32(u[0])
                aux["mask_y_u1"] = np.int64(1)
            if target_view == 1:
                x1 = x.astype(np.float32)
            elif target_view == 2:
                x2 = x.astype(np.float32)
            else:
                x3 = x.astype(np.float32)

        elif name in ("R12", "R13", "R23"):
            rho = self._sample_rho()
            r = self.rng.normal(size=(self.cfg.m,))
            eta_i = self.rng.normal(size=(self.cfg.m,))
            eta_j = self.rng.normal(size=(self.cfg.m,))
            ri = np.sqrt(rho) * r + np.sqrt(1.0 - rho) * eta_i
            rj = np.sqrt(rho) * r + np.sqrt(1.0 - rho) * eta_j
            i, j = int(name[1]), int(name[2])
            xi = self._project(i, name, ri, alpha)
            xj = self._project(j, name, rj, alpha)
            if return_aux and pid_id == 3:
                aux["y_r12"] = np.float32(r[0])
                aux["mask_y_r12"] = np.int64(1)
            if i == 1:
                x1 = xi.astype(np.float32)
            elif i == 2:
                x2 = xi.astype(np.float32)
            else:
                x3 = xi.astype(np.float32)
            if j == 1:
                x1 = xj.astype(np.float32)
            elif j == 2:
                x2 = xj.astype(np.float32)
            else:
                x3 = xj.astype(np.float32)

        elif name == "R123":
            rho = self._sample_rho()
            r = self.rng.normal(size=(self.cfg.m,))
            etas = self.rng.normal(size=(3, self.cfg.m))
            rs = [np.sqrt(rho) * r + np.sqrt(1.0 - rho) * etas[k] for k in range(3)]
            x1 = self._project(1, "R123", rs[0], alpha).astype(np.float32)
            x2 = self._project(2, "R123", rs[1], alpha).astype(np.float32)
            x3 = self._project(3, "R123", rs[2], alpha).astype(np.float32)
            if return_aux:
                aux["y_r123"] = np.float32(r[0])
                aux["mask_y_r123"] = np.int64(1)

        elif name in ("S12->3", "S13->2", "S23->1"):
            pair = name[1:3]
            target = int(name[-1])
            hop = self._sample_hop()
            a = self.rng.normal(size=(self.cfg.m,))
            b = self.rng.normal(size=(self.cfg.m,))
            source_i, source_j = int(pair[0]), int(pair[1])
            comp_a = f"A{pair}"
            comp_b = f"B{pair}"
            syn_comp = f"SYN{pair}"
            xi = self._project(source_i, comp_a, a, alpha)
            xj = self._project(source_j, comp_b, b, alpha)
            s0 = self.synergy_mlp(a, b, hop)
            s = s0 - (a @ self.C_a[hop]) - (b @ self.C_b[hop])
            xk = self._project(target, syn_comp, s, alpha)
            if return_aux and pid_id == 7:
                aux["y_s12_3"] = np.float32(s[0])
                aux["mask_y_s12_3"] = np.int64(1)

            if source_i == 1:
                x1 = xi.astype(np.float32)
            elif source_i == 2:
                x2 = xi.astype(np.float32)
            else:
                x3 = xi.astype(np.float32)

            if source_j == 1:
                x1 = xj.astype(np.float32)
            elif source_j == 2:
                x2 = xj.astype(np.float32)
            else:
                x3 = xj.astype(np.float32)

            if target == 1:
                x1 = xk.astype(np.float32)
            elif target == 2:
                x2 = xk.astype(np.float32)
            else:
                x3 = xk.astype(np.float32)

        else:
            raise RuntimeError(f"Unhandled PID atom: {name}")

        x1 = (x1 + self._noise()).astype(np.float32)
        x2 = (x2 + self._noise()).astype(np.float32)
        x3 = (x3 + self._noise()).astype(np.float32)

        out = {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "pid_id": np.int64(pid_id),
            "alpha": np.float32(alpha),
            "sigma": np.float32(sigma),
            "rho": np.float32(rho),
            "hop": np.int64(hop),
        }
        if return_aux:
            out.update(aux)
        return out

    def generate(
        self,
        n: int,
        pid_ids: Optional[Sequence[int]] = None,
        return_aux: bool = False,
    ) -> Dict[str, np.ndarray]:
        if pid_ids is not None and len(pid_ids) != n:
            raise ValueError("If pid_ids is provided, its length must match n")

        out = {
            "x1": np.zeros((n, self.cfg.d), dtype=np.float32),
            "x2": np.zeros((n, self.cfg.d), dtype=np.float32),
            "x3": np.zeros((n, self.cfg.d), dtype=np.float32),
            "pid_id": np.zeros((n,), dtype=np.int64),
            "alpha": np.zeros((n,), dtype=np.float32),
            "sigma": np.zeros((n,), dtype=np.float32),
            "rho": np.zeros((n,), dtype=np.float32),
            "hop": np.zeros((n,), dtype=np.int64),
        }
        if return_aux:
            out.update(
                {
                    "y_u1": np.zeros((n,), dtype=np.float32),
                    "mask_y_u1": np.zeros((n,), dtype=np.int64),
                    "y_r12": np.zeros((n,), dtype=np.float32),
                    "mask_y_r12": np.zeros((n,), dtype=np.int64),
                    "y_r123": np.zeros((n,), dtype=np.float32),
                    "mask_y_r123": np.zeros((n,), dtype=np.int64),
                    "y_s12_3": np.zeros((n,), dtype=np.float32),
                    "mask_y_s12_3": np.zeros((n,), dtype=np.int64),
                }
            )

        for idx in range(n):
            sample = self.sample(None if pid_ids is None else int(pid_ids[idx]), return_aux=return_aux)
            for key in out:
                out[key][idx] = sample[key]
        return out


def pid_name(pid_id: int) -> str:
    return PID_ID_TO_NAME[int(pid_id)]


def all_pid_names() -> List[str]:
    return [PID_ID_TO_NAME[i] for i in range(10)]
