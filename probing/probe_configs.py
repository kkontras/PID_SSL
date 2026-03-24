"""Probe input combinations: which node representations to combine for probing."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def get_probe_inputs(
    z1: np.ndarray, z2: np.ndarray, z3: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Returns all 7 standard probe input combinations.
    z1, z2, z3: (N, d) representation arrays.
    """
    return {
        "z1":       z1,
        "z2":       z2,
        "z3":       z3,
        "z1_z2":    np.concatenate([z1, z2], axis=-1),
        "z1_z3":    np.concatenate([z1, z3], axis=-1),
        "z2_z3":    np.concatenate([z2, z3], axis=-1),
        "z1_z2_z3": np.concatenate([z1, z2, z3], axis=-1),
    }


def get_confu_probe_inputs(
    z1: np.ndarray, z2: np.ndarray, z3: np.ndarray,
    f12: np.ndarray, f13: np.ndarray, f23: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extended probe inputs for ConFu, including fusion representations."""
    base = get_probe_inputs(z1, z2, z3)
    base.update({"f12": f12, "f13": f13, "f23": f23})
    return base
