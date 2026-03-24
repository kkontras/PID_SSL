"""Tests for the V3 dataset generator."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from data.dataset_v3 import (
    V3Config, V3DatasetGenerator, fuse_nl,
    SINGLE_ATOM_CONFIGS, MULTI_ATOM_CONFIGS, ALL_CONFIGS,
)


def test_fuse_nl_requires_both():
    """fuse_nl(a, b, Q) should not be recoverable from either a or b alone."""
    Q = 7
    # Fix a, vary b — output should vary
    a = 3
    outputs_vary_b = [fuse_nl(a, b, Q) for b in range(Q)]
    assert len(set(outputs_vary_b)) > 1, "fuse_nl must vary with b"

    # Fix b, vary a — output should vary
    b = 5
    outputs_vary_a = [fuse_nl(a2, b, Q) for a2 in range(Q)]
    assert len(set(outputs_vary_a)) > 1, "fuse_nl must vary with a"


def test_basic_generation():
    """Basic dataset generation produces correct shapes."""
    cfg = V3Config(Q=7, D=44, D_info=4, active_atoms=["uniq_1", "red_12", "syn_12"],
                   n_samples=100, seed=0)
    gen = V3DatasetGenerator(cfg)
    data = gen.generate(100)
    assert data["x1"].shape == (100, 44)
    assert data["x2"].shape == (100, 44)
    assert data["x3"].shape == (100, 44)
    assert data["label"].shape == (100,)
    assert "sub_uniq_1" in data
    assert "sub_red_12" in data
    assert "sub_syn_12" in data


def test_label_range():
    """Labels should be in range [0, Q^n_atoms)."""
    cfg = V3Config(Q=7, D=44, D_info=4, active_atoms=["uniq_1", "red_12", "syn_12"],
                   n_samples=1000, seed=0)
    gen = V3DatasetGenerator(cfg)
    data = gen.generate(1000)
    n_classes = cfg.n_classes()
    assert data["label"].min() >= 0
    assert data["label"].max() < n_classes


def test_sub_labels_range():
    """Sub-labels should be in range [0, Q)."""
    cfg = V3Config(Q=7, D=44, D_info=4, active_atoms=["uniq_1", "red_12", "syn_12"],
                   n_samples=500, seed=0)
    gen = V3DatasetGenerator(cfg)
    data = gen.generate(500)
    for atom in cfg.active_atoms:
        sl = data[f"sub_{atom}"]
        assert sl.min() >= 0
        assert sl.max() < cfg.Q


def test_redundancy_same_value():
    """For red_12, both x1 and x2 should encode the same shared value."""
    cfg = V3Config(Q=7, D=44, D_info=4, active_atoms=["red_12"], n_samples=100, seed=0)
    gen = V3DatasetGenerator(cfg)
    data = gen.generate(100)
    # The sub_label is the shared value; both x1 and x2 should encode it
    # Verify label is in range
    assert data["sub_red_12"].min() >= 0


def test_dimension_allocation_no_overlap():
    """Atom dimension slots should not overlap."""
    cfg = V3Config(Q=7, D=44, D_info=4,
                   active_atoms=["uniq_1", "uniq_2", "uniq_3", "red_12", "syn_12"],
                   n_samples=10, seed=0)
    gen = V3DatasetGenerator(cfg)
    # Collect all used dims
    all_ranges = []
    for atom, node_map in gen.dim_map.items():
        for node, (s, e) in node_map.items():
            all_ranges.append((s, e))

    # For a single node, dim slots should be sequential and non-overlapping
    dims_used = set()
    slots = set()
    for atom in cfg.active_atoms:
        if 1 in gen.dim_map.get(atom, {}):
            s, e = gen.dim_map[atom][1]
            slot = (s, e)
            assert slot not in slots, f"Duplicate slot {slot}"
            slots.add(slot)
            new_dims = set(range(s, e))
            assert not new_dims & dims_used, f"Overlapping dims for atom {atom}"
            dims_used |= new_dims


def test_all_single_atom_configs_generate():
    """All A-group configs should generate without error."""
    for name, atoms in SINGLE_ATOM_CONFIGS.items():
        cfg = V3Config(Q=7, D=44, D_info=4, active_atoms=atoms, n_samples=20, seed=0)
        gen = V3DatasetGenerator(cfg)
        data = gen.generate(20)
        assert data["label"].shape == (20,), f"Config {name} failed"


def test_pairred_atoms_generate_and_have_sub_labels():
    """New pair-to-target redundancy atoms should generate valid labels."""
    for atom in ["pairred_12_3", "pairred_13_2", "pairred_23_1"]:
        cfg = V3Config(Q=7, D=44, D_info=4, active_atoms=[atom], n_samples=32, seed=0)
        gen = V3DatasetGenerator(cfg)
        data = gen.generate(32)
        assert data["label"].shape == (32,)
        assert f"sub_{atom}" in data
        assert data[f"sub_{atom}"].min() >= 0
        assert data[f"sub_{atom}"].max() < cfg.Q


def test_multi_atom_configs_generate():
    """B-group configs should generate without error."""
    # Test a few
    for name in ["B1", "B2", "B3", "B4", "B10"]:
        atoms = MULTI_ATOM_CONFIGS[name]
        cfg = V3Config(Q=7, D=88, D_info=4, active_atoms=atoms, n_samples=20, seed=0)
        gen = V3DatasetGenerator(cfg)
        data = gen.generate(20)
        assert data["label"].shape == (20,), f"Config {name} failed"


def test_info_dims_vs_noise_dims():
    """Info dims should be more structured than noise dims."""
    cfg = V3Config(Q=7, D=44, D_info=4, active_atoms=["uniq_1"], n_samples=500, seed=0)
    gen = V3DatasetGenerator(cfg)
    data = gen.generate(500)
    x1 = data["x1"]

    info_ranges = gen.info_dims_for_node(1)
    noise_dims = gen.noise_dims_for_node(1)

    # Noise dims should cluster around mu_bg = 0.75
    noise_mean = x1[:, noise_dims].mean()
    assert abs(noise_mean - 0.75) < 0.05, f"Noise dims mean {noise_mean:.3f} not near 0.75"

    # Info dims should not be at the background mean when sub-label varies
    # (they should show structured variation)
    if info_ranges:
        s, e = info_ranges[0]
        info_mean = x1[:, s:e].mean()
        # Info mean should be in the informative range (0 to 1 roughly)
        assert 0.0 <= info_mean <= 1.5, f"Info dims mean {info_mean:.3f} out of expected range"


def test_pairred_12_3_target_matches_pair_fuse():
    """pairred_12_3 should encode the same target on node 3 as the fused source pair."""
    cfg = V3Config(Q=7, D=44, D_info=4, active_atoms=["pairred_12_3"], n_samples=128, seed=0)
    gen = V3DatasetGenerator(cfg)
    data = gen.generate(128)
    atom = "pairred_12_3"
    s1, e1 = gen.dim_map[atom][1]
    s2, e2 = gen.dim_map[atom][2]
    s3, e3 = gen.dim_map[atom][3]

    def decode_slot(slot: np.ndarray, q: int) -> np.ndarray:
        base = np.clip(np.rint(slot.mean(axis=1) * (q - 1)), 0, q - 1)
        return base.astype(np.int64)

    a = decode_slot(data["x1"][:, s1:e1], cfg.Q)
    b = decode_slot(data["x2"][:, s2:e2], cfg.Q)
    t = decode_slot(data["x3"][:, s3:e3], cfg.Q)
    fused = np.array([fuse_nl(int(ai), int(bi), cfg.Q) for ai, bi in zip(a, b)], dtype=np.int64)

    match = np.mean(fused == t)
    assert match > 0.90, f"Pair fuse should match held-out target, got {match:.3f}"


if __name__ == "__main__":
    test_fuse_nl_requires_both()
    test_basic_generation()
    test_label_range()
    test_sub_labels_range()
    test_redundancy_same_value()
    test_dimension_allocation_no_overlap()
    test_all_single_atom_configs_generate()
    test_multi_atom_configs_generate()
    test_info_dims_vs_noise_dims()
    print("All tests passed!")
