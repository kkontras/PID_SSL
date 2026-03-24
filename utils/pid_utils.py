"""PID atom definitions and dimension allocation utilities."""
from __future__ import annotations

from typing import Dict, List, Tuple

from data.dataset_v3 import ATOM_NAMES, SINGLE_ATOM_CONFIGS, MULTI_ATOM_CONFIGS, ASYMMETRIC_CONFIGS, ALL_CONFIGS


def atom_type(atom: str) -> str:
    """Returns 'unique', 'redundant', 'synergistic', or 'pair_redundant'."""
    if atom.startswith("uniq_"):
        return "unique"
    if atom.startswith("red_"):
        return "redundant"
    if atom.startswith("syn_"):
        return "synergistic"
    if atom.startswith("pairred_"):
        return "pair_redundant"
    raise ValueError(f"Unknown atom: {atom}")


def atoms_for_node(node: int, active_atoms: List[str]) -> List[str]:
    """Return all active atoms that involve the given node (1-based)."""
    from data.dataset_v3 import V3DatasetGenerator
    # Use node lookup from generator
    result = []
    for atom in active_atoms:
        if atom.startswith("uniq_"):
            if int(atom[-1]) == node:
                result.append(atom)
        elif atom in ("red_12", "syn_12") and node in (1, 2):
            result.append(atom)
        elif atom in ("red_13", "syn_13") and node in (1, 3):
            result.append(atom)
        elif atom in ("red_23", "syn_23") and node in (2, 3):
            result.append(atom)
        elif atom in ("red_123", "syn_123", "pairred_12_3", "pairred_13_2", "pairred_23_1"):
            result.append(atom)
    return result


def config_info(config_name: str) -> Dict:
    """Return metadata about a named config."""
    atoms = ALL_CONFIGS[config_name]
    return {
        "config_name": config_name,
        "atoms": atoms,
        "n_atoms": len(atoms),
        "unique_atoms": [a for a in atoms if a.startswith("uniq_")],
        "redundant_atoms": [a for a in atoms if a.startswith("red_")],
        "synergistic_atoms": [a for a in atoms if a.startswith("syn_")],
        "pair_redundant_atoms": [a for a in atoms if a.startswith("pairred_")],
    }
