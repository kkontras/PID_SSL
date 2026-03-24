"""Visualization utilities: heatmaps, radar charts, learning curves."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def print_probe_table(results: Dict[str, Dict[str, float]], title: str = "Probe Results") -> None:
    """Print a markdown-style table of probe results."""
    if not results:
        return
    methods = list(results.keys())
    configs = list(results[methods[0]].keys())
    col_w = max(max(len(c) for c in configs), 10)
    meth_w = max(max(len(m) for m in methods), 8)

    print(f"\n### {title}")
    header = f"{'Method':<{meth_w}} | " + " | ".join(f"{c:>{col_w}}" for c in configs)
    print(header)
    print("-" * len(header))
    for method in methods:
        row = f"{method:<{meth_w}} | "
        row += " | ".join(
            f"{results[method].get(c, float('nan')):>{col_w}.3f}" for c in configs
        )
        print(row)


def print_per_atom_table(
    results: Dict[str, Dict[str, float]],
    atom_names: List[str],
    title: str = "Per-Atom Accuracy",
) -> None:
    """Print per-atom accuracy table."""
    methods = list(results.keys())
    col_w = max(max(len(a) for a in atom_names), 10)
    meth_w = max(max(len(m) for m in methods), 8)

    print(f"\n### {title}")
    header = f"{'Method':<{meth_w}} | " + " | ".join(f"{a:>{col_w}}" for a in atom_names)
    print(header)
    print("-" * len(header))
    for method in methods:
        row = f"{method:<{meth_w}} | "
        row += " | ".join(
            f"{results[method].get(a, float('nan')):>{col_w}.3f}" for a in atom_names
        )
        print(row)
