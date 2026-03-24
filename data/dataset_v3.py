"""V3 PID dataset generator: discrete integer latents, modular-arithmetic labels, explicit atom-slot feature encoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Atom identifiers ───────────────────────────────────────────────────────────
ATOM_NAMES = [
    "uniq_1", "uniq_2", "uniq_3",
    "red_12", "red_13", "red_23", "red_123",
    "syn_12", "syn_13", "syn_23", "syn_123",
    "pairred_12_3", "pairred_13_2", "pairred_23_1",
]

# Single-atom dataset configs (A-group)
SINGLE_ATOM_CONFIGS: Dict[str, List[str]] = {
    "A1":  ["uniq_1"],
    "A2":  ["uniq_2"],
    "A3":  ["uniq_3"],
    "A4":  ["red_12"],
    "A5":  ["red_13"],
    "A6":  ["red_23"],
    "A7":  ["red_123"],
    "A8":  ["syn_12"],
    "A9":  ["syn_13"],
    "A10": ["syn_23"],
    "A11": ["syn_123"],
    "A12": ["pairred_12_3"],
    "A13": ["pairred_13_2"],
    "A14": ["pairred_23_1"],
}

# Multi-atom dataset configs (B-group)
MULTI_ATOM_CONFIGS: Dict[str, List[str]] = {
    "B1":  ["uniq_1", "uniq_2", "uniq_3"],
    "B2":  ["red_12", "red_13", "red_23"],
    "B3":  ["syn_12", "syn_13", "syn_23"],
    "B4":  ["uniq_1", "red_12", "syn_12"],
    "B5":  ["uniq_1", "uniq_2", "red_12", "syn_12"],
    "B6":  ["red_12", "red_13", "red_23", "red_123"],
    "B7":  ["uniq_1", "red_12", "syn_23"],
    "B8":  ["uniq_1", "uniq_2", "uniq_3", "red_12", "red_13", "red_23", "syn_12"],
    "B9":  ["syn_12", "syn_13", "syn_123"],
    "B10": ["red_123", "syn_123"],
}

# Asymmetric / stress-test configs (C-group)
ASYMMETRIC_CONFIGS: Dict[str, List[str]] = {
    "C2": ["syn_12", "syn_13", "syn_23"],
    "C3": ["red_12", "syn_12"],
}

ALL_CONFIGS: Dict[str, List[str]] = {**SINGLE_ATOM_CONFIGS, **MULTI_ATOM_CONFIGS, **ASYMMETRIC_CONFIGS}


def _make_pair_synergy_table(Q: int, seed: int) -> np.ndarray:
    """
    Build a balanced pairwise lookup table T[a, b] -> y.

    We use a randomized Latin-square construction so every row and every column
    contains every output class exactly once. This keeps the pairwise synergy
    target balanced while avoiding a fixed hand-coded arithmetic fuse.
    """
    rng = np.random.default_rng(seed)
    row_perm = rng.permutation(Q)
    col_perm = rng.permutation(Q)
    symbol_perm = rng.permutation(Q)
    table = np.zeros((Q, Q), dtype=np.int64)
    for a in range(Q):
        for b in range(Q):
            table[a, b] = symbol_perm[(row_perm[a] + col_perm[b]) % Q]
    return table


def _make_triple_synergy_perms(Q: int, seed: int) -> np.ndarray:
    """
    Build one symbol permutation per c-value so T[a,b,c] = P_c[T_pair[a,b]].
    This keeps the triple target dependent on all three inputs without relying
    on nested modular arithmetic.
    """
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(Q) for _ in range(Q)], axis=0).astype(np.int64)


@dataclass
class V3Config:
    Q: int = 7                    # modular arithmetic base (prime)
    D: int = 24                   # total feature dimension per node
    D_info: int = 4               # feature dimensions per atom slot
    n_nodes: int = 3              # always 3
    active_atoms: List[str] = field(default_factory=lambda: ["uniq_1", "uniq_2", "uniq_3", "red_12", "syn_12"])
    sigma_info: float = 0.01      # noise std on informative dims
    mu_bg: float = 0.75           # background mean for non-informative dims
    sigma_bg: float = 0.3         # background std
    n_samples: int = 50000
    seed: int = 42

    def n_classes(self) -> int:
        return self.Q ** len(self.active_atoms)


class V3DatasetGenerator:
    """
    V3 PID dataset: discrete integer latents, modular-arithmetic labels,
    explicit per-atom dimension allocation.

    Each sample has:
      - x1, x2, x3  in R^D
      - label Y  in {0,...,Q^n_atoms - 1} (composite integer)
      - sub_labels  dict mapping atom name -> integer in {0,...,Q-1}

    Feature layout:
      - Informative dims encode integer values as scaled linear codes
      - Non-informative dims are Gaussian background noise
    """

    def __init__(self, cfg: Optional[V3Config] = None):
        self.cfg = cfg or V3Config()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._pair_synergy_table = _make_pair_synergy_table(self.cfg.Q, self.cfg.seed + 17)
        self._triple_synergy_perms = _make_triple_synergy_perms(self.cfg.Q, self.cfg.seed + 29)
        self._dim_map = self._build_dim_map()

    def _fuse_pair(self, a: int, b: int) -> int:
        return int(self._pair_synergy_table[a, b])

    def _fuse_triple(self, a: int, b: int, c: int) -> int:
        pair_val = self._fuse_pair(a, b)
        return int(self._triple_synergy_perms[c, pair_val])

    # ── Dimension allocation ───────────────────────────────────────────────────

    def _build_dim_map(self) -> Dict[str, Dict[int, Tuple[int, int]]]:
        """
        Returns dim_map[atom][(node)] = (dim_start, dim_end).
        Atoms are allocated sequentially in D_info-sized slots.
        If total required dims exceed D, raises an error.
        """
        D_info = self.cfg.D_info
        active = self.cfg.active_atoms
        # Check capacity
        required = len(active) * D_info
        if required > self.cfg.D:
            raise ValueError(
                f"Not enough feature dimensions: need {required} (= {len(active)} atoms × {D_info} dims/atom) "
                f"but D={self.cfg.D}. Increase D or decrease D_info or active_atoms."
            )

        dim_map: Dict[str, Dict[int, Tuple[int, int]]] = {}
        slot = 0
        for atom in active:
            d_start = slot * D_info
            d_end = d_start + D_info
            slot += 1
            nodes_for_atom = self._nodes_for_atom(atom)
            dim_map[atom] = {node: (d_start, d_end) for node in nodes_for_atom}
        return dim_map

    @staticmethod
    def _nodes_for_atom(atom: str) -> List[int]:
        """Return which node indices (1-based) participate in this atom."""
        if atom == "uniq_1":
            return [1]
        if atom == "uniq_2":
            return [2]
        if atom == "uniq_3":
            return [3]
        if atom == "red_12":
            return [1, 2]
        if atom == "red_13":
            return [1, 3]
        if atom == "red_23":
            return [2, 3]
        if atom == "red_123":
            return [1, 2, 3]
        if atom == "syn_12":
            return [1, 2]  # each node carries one independent part
        if atom == "syn_13":
            return [1, 3]
        if atom == "syn_23":
            return [2, 3]
        if atom == "syn_123":
            return [1, 2, 3]
        if atom == "pairred_12_3":
            return [1, 2, 3]
        if atom == "pairred_13_2":
            return [1, 2, 3]
        if atom == "pairred_23_1":
            return [1, 2, 3]
        raise ValueError(f"Unknown atom: {atom}")

    # ── Feature encoding ───────────────────────────────────────────────────────

    def _encode_integer(self, z: int, dim_start: int, dim_end: int, out: np.ndarray) -> None:
        """Linear encoding of integer z in [0,Q-1] into out[dim_start:dim_end]."""
        Q = self.cfg.Q
        D_info = dim_end - dim_start
        base = z / max(Q - 1, 1)
        for d in range(D_info):
            s_d = 1.0 + 0.5 * d / max(D_info - 1, 1)
            o_d = 0.1 * d / max(D_info - 1, 1)
            noise = self.rng.normal(0.0, self.cfg.sigma_info)
            out[dim_start + d] = base * s_d + o_d + noise

    # ── Label computation ──────────────────────────────────────────────────────

    def _compute_sub_label(
        self,
        atom: str,
        v: Dict[int, int],         # node index (1-based) -> unique integer
        r: Dict[str, int],         # atom -> shared redundancy integer
        s: Dict[str, Dict[int, int]],  # atom -> node -> synergy part integer
    ) -> int:
        Q = self.cfg.Q
        if atom == "uniq_1":
            return v[1]
        if atom == "uniq_2":
            return v[2]
        if atom == "uniq_3":
            return v[3]
        if atom == "red_12":
            return r["red_12"]
        if atom == "red_13":
            return r["red_13"]
        if atom == "red_23":
            return r["red_23"]
        if atom == "red_123":
            return r["red_123"]
        if atom == "syn_12":
            return self._fuse_pair(s["syn_12"][1], s["syn_12"][2])
        if atom == "syn_13":
            return self._fuse_pair(s["syn_13"][1], s["syn_13"][3])
        if atom == "syn_23":
            return self._fuse_pair(s["syn_23"][2], s["syn_23"][3])
        if atom == "syn_123":
            return self._fuse_triple(s["syn_123"][1], s["syn_123"][2], s["syn_123"][3])
        if atom == "pairred_12_3":
            return s["pairred_12_3"][3]
        if atom == "pairred_13_2":
            return s["pairred_13_2"][2]
        if atom == "pairred_23_1":
            return s["pairred_23_1"][1]
        raise ValueError(f"Unknown atom: {atom}")

    # ── Sample generation ──────────────────────────────────────────────────────

    def _sample_one(self) -> Dict:
        Q = self.cfg.Q
        D = self.cfg.D
        active = self.cfg.active_atoms

        # Sample node-unique integers
        v: Dict[int, int] = {i: int(self.rng.integers(0, Q)) for i in range(1, 4)}

        # Sample shared redundancy integers
        r: Dict[str, int] = {}
        for atom in active:
            if atom.startswith("red_"):
                r[atom] = int(self.rng.integers(0, Q))

        # Sample synergy part integers (one per participating node)
        s: Dict[str, Dict[int, int]] = {}
        for atom in active:
            if atom.startswith("syn_") or atom.startswith("pairred_"):
                nodes = self._nodes_for_atom(atom)
                s[atom] = {node: int(self.rng.integers(0, Q)) for node in nodes}

        # Reparameterize pair-to-target redundancy so the source pair determines
        # the held-out target label via the same pairwise synergy lookup table.
        for atom in active:
            if atom == "pairred_12_3":
                s[atom][3] = self._fuse_pair(s[atom][1], s[atom][2])
            elif atom == "pairred_13_2":
                s[atom][2] = self._fuse_pair(s[atom][1], s[atom][3])
            elif atom == "pairred_23_1":
                s[atom][1] = self._fuse_pair(s[atom][2], s[atom][3])

        # Initialize features with background noise
        x = {1: self.rng.normal(self.cfg.mu_bg, self.cfg.sigma_bg, size=(D,)).astype(np.float32),
             2: self.rng.normal(self.cfg.mu_bg, self.cfg.sigma_bg, size=(D,)).astype(np.float32),
             3: self.rng.normal(self.cfg.mu_bg, self.cfg.sigma_bg, size=(D,)).astype(np.float32)}

        # Encode each active atom into the relevant node dimension slots
        for atom in active:
            if atom.startswith("uniq_"):
                node = int(atom[-1])
                d_start, d_end = self._dim_map[atom][node]
                self._encode_integer(v[node], d_start, d_end, x[node])

            elif atom.startswith("red_"):
                nodes = self._nodes_for_atom(atom)
                shared_val = r[atom]
                for node in nodes:
                    d_start, d_end = self._dim_map[atom][node]
                    # All nodes encode same shared_val (redundancy)
                    self._encode_integer(shared_val, d_start, d_end, x[node])

            elif atom.startswith("syn_"):
                nodes = self._nodes_for_atom(atom)
                for node in nodes:
                    d_start, d_end = self._dim_map[atom][node]
                    # Each node encodes its own independent part
                    self._encode_integer(s[atom][node], d_start, d_end, x[node])

            elif atom.startswith("pairred_"):
                if atom == "pairred_12_3":
                    self._encode_integer(s[atom][1], *self._dim_map[atom][1], x[1])
                    self._encode_integer(s[atom][2], *self._dim_map[atom][2], x[2])
                    self._encode_integer(s[atom][3], *self._dim_map[atom][3], x[3])
                elif atom == "pairred_13_2":
                    self._encode_integer(s[atom][1], *self._dim_map[atom][1], x[1])
                    self._encode_integer(s[atom][3], *self._dim_map[atom][3], x[3])
                    self._encode_integer(s[atom][2], *self._dim_map[atom][2], x[2])
                elif atom == "pairred_23_1":
                    self._encode_integer(s[atom][2], *self._dim_map[atom][2], x[2])
                    self._encode_integer(s[atom][3], *self._dim_map[atom][3], x[3])
                    self._encode_integer(s[atom][1], *self._dim_map[atom][1], x[1])

        # Compute sub-labels and composite label
        sub_labels: Dict[str, int] = {}
        for k, atom in enumerate(active):
            sub_labels[atom] = self._compute_sub_label(atom, v, r, s)

        # Composite label: Y = sum_k sub_label[k] * Q^k
        label = 0
        for k, atom in enumerate(active):
            label += sub_labels[atom] * (Q ** k)

        return {
            "x1": x[1].astype(np.float32),
            "x2": x[2].astype(np.float32),
            "x3": x[3].astype(np.float32),
            "label": np.int64(label),
            "sub_labels": {atom: np.int64(sub_labels[atom]) for atom in active},
        }

    def generate(self, n: int) -> Dict[str, np.ndarray]:
        """Generate n samples. Returns dict of stacked arrays."""
        samples = [self._sample_one() for _ in range(n)]
        active = self.cfg.active_atoms
        out: Dict[str, np.ndarray] = {
            "x1": np.stack([s["x1"] for s in samples]),
            "x2": np.stack([s["x2"] for s in samples]),
            "x3": np.stack([s["x3"] for s in samples]),
            "label": np.array([s["label"] for s in samples], dtype=np.int64),
        }
        for atom in active:
            out[f"sub_{atom}"] = np.array([s["sub_labels"][atom] for s in samples], dtype=np.int64)
        return out

    def generate_split(self, n_train: int, n_val: int, n_test: int) -> Tuple[Dict, Dict, Dict]:
        """Generate train/val/test splits."""
        train = self.generate(n_train)
        val = self.generate(n_val)
        test = self.generate(n_test)
        return train, val, test

    @property
    def dim_map(self) -> Dict[str, Dict[int, Tuple[int, int]]]:
        return self._dim_map

    def info_dims_for_node(self, node: int) -> List[Tuple[int, int]]:
        """Return list of (start, end) informative dim ranges for a given node."""
        ranges = []
        for atom, node_map in self._dim_map.items():
            if node in node_map:
                ranges.append(node_map[node])
        return ranges

    def noise_dims_for_node(self, node: int) -> List[int]:
        """Return list of dimension indices that are background noise for this node."""
        info_set = set()
        for start, end in self.info_dims_for_node(node):
            info_set.update(range(start, end))
        return [d for d in range(self.cfg.D) if d not in info_set]


def make_generator(config_name: str, Q: int = 7, D: int = 44, D_info: int = 4,
                   n_samples: int = 50000, seed: int = 42) -> V3DatasetGenerator:
    """Convenience factory: create generator from a named config (A1-A14, B1-B10, C2-C3)."""
    atoms = ALL_CONFIGS[config_name]
    cfg = V3Config(Q=Q, D=D, D_info=D_info, active_atoms=atoms, n_samples=n_samples, seed=seed)
    return V3DatasetGenerator(cfg)
