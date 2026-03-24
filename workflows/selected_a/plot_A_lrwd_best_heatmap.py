"""
plot_A_lrwd_best_heatmap.py
----------------------------
For each (config A1-A14, method) pair, selects the best (lr, wd) hyperparameter
combination from the grid search based on minimum validation loss across all
training epochs, then reads the corresponding probe accuracy (z1_z2_z3) and
renders a heatmap grouped by PID type.

Usage:
    python plot_A_lrwd_best_heatmap.py
"""

import csv
import json
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOT = REPO_ROOT / "test_outputs" / "v3_runs_A_lrwd_search"
OUT_PATH_BASE = SEARCH_ROOT / "best_tuned_heatmap.png"
OUT_PATH_ALL = SEARCH_ROOT / "best_tuned_heatmap_all_methods.png"

# ---------------------------------------------------------------------------
# Config & method metadata
# ---------------------------------------------------------------------------
CONFIGS = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14"]
BASE_METHODS = ["simclr", "pairwise_nce", "triangle", "confu"]
ALL_METHODS = ["simclr", "pairwise_nce", "triangle", "confu", "masked_raw", "masked_emb", "comm", "infmask"]

CONFIG_DISPLAY = {
    "A1":  "Unique X\u2081",
    "A2":  "Unique X\u2082",
    "A3":  "Unique X\u2083",
    "A4":  "Redundant X\u2081=X\u2082",
    "A5":  "Redundant X\u2081=X\u2083",
    "A6":  "Redundant X\u2082=X\u2083",
    "A7":  "Redundant X\u2081=X\u2082=X\u2083",
    "A8":  "Synergy X\u2081\u2295X\u2082",
    "A9":  "Synergy X\u2081\u2295X\u2083",
    "A10": "Synergy X\u2082\u2295X\u2083",
    "A11": "Synergy X\u2081\u2295X\u2082\u2295X\u2083",
    "A12": "Pair-Red (X\u2081,X\u2082)\u2192X\u2083",
    "A13": "Pair-Red (X\u2081,X\u2083)\u2192X\u2082",
    "A14": "Pair-Red (X\u2082,X\u2083)\u2192X\u2081",
}

METHOD_DISPLAY = {
    "simclr":       "SimCLR",
    "pairwise_nce": "Pairwise NCE",
    "triangle":     "TRIANGLE",
    "confu":        "ConFu",
    "masked_raw":   "Masked Raw",
    "masked_emb":   "Masked Emb",
    "comm":         "CoMM",
    "infmask":      "InfMasking",
}

# PID type groups: (label, list of configs, chance accuracy)
GROUPS = [
    ("Unique",     ["A1","A2","A3"],               1/7),
    ("Redundant",  ["A4","A5","A6","A7"],           1/7),
    ("Synergy",    ["A8","A9","A10","A11"],         1/49),
    ("Pair-Red",   ["A12","A13","A14"],             1/49),
]


# ---------------------------------------------------------------------------
# Step 1 & 2: Load best-tuned results
# ---------------------------------------------------------------------------

def _read_min_val_loss(run_dir: Path) -> float:
    """Return the minimum val_loss across all epochs for a run directory."""
    hist_csv = run_dir / "pretrain" / "history.csv"
    if not hist_csv.exists():
        return float("inf")
    min_loss = float("inf")
    with open(hist_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row["val_loss"])
                if v < min_loss:
                    min_loss = v
            except (KeyError, ValueError):
                pass
    return min_loss


def _read_probe_acc(run_dir: Path) -> Optional[float]:
    """Return overall z1_z2_z3 probe accuracy from probe_results.json."""
    probe_json = run_dir / "probe" / "probe_results.json"
    if not probe_json.exists():
        return None
    with open(probe_json) as f:
        data = json.load(f)
    try:
        return float(data["overall"]["z1_z2_z3"])
    except (KeyError, TypeError):
        return None


def load_best_results(methods: List[str]) -> dict:
    """
    Returns a dict: results[(config, method)] = {
        "probe_acc": float or None,
        "best_lr": str,
        "best_wd": str,
        "min_val_loss": float,
    }
    """
    results = {}
    for config in CONFIGS:
        for method in methods:
            method_dir = SEARCH_ROOT / config / method
            if not method_dir.exists():
                results[(config, method)] = {"probe_acc": None, "best_lr": "?", "best_wd": "?", "min_val_loss": float("inf")}
                continue

            best_loss = float("inf")
            best_run_dir = None
            best_lr = "?"
            best_wd = "?"

            # Each subdir is lr_X__wd_Y/seed_101
            for hp_dir in sorted(method_dir.iterdir()):
                if not hp_dir.is_dir():
                    continue
                seed_dir = hp_dir / "seed_101"
                if not seed_dir.exists():
                    # Try any seed dir
                    seed_dirs = [d for d in hp_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
                    seed_dir = seed_dirs[0] if seed_dirs else None
                if seed_dir is None:
                    continue
                loss = _read_min_val_loss(seed_dir)
                if loss < best_loss:
                    best_loss = loss
                    best_run_dir = seed_dir
                    # Parse lr/wd from dir name e.g. "lr_1e-3__wd_1e-5"
                    parts = hp_dir.name.split("__")
                    best_lr = parts[0].replace("lr_", "") if len(parts) > 0 else "?"
                    best_wd = parts[1].replace("wd_", "") if len(parts) > 1 else "?"

            probe_acc = _read_probe_acc(best_run_dir) if best_run_dir else None
            results[(config, method)] = {
                "probe_acc": probe_acc,
                "best_lr": best_lr,
                "best_wd": best_wd,
                "min_val_loss": best_loss,
            }

    return results


# ---------------------------------------------------------------------------
# Step 3-7: Build matrix & plot
# ---------------------------------------------------------------------------

def print_summary_table(results: dict, methods: List[str], title: str):
    print(title)
    print(f"\n{'Config':<6} {'Method':<14} {'Best LR':<8} {'Best WD':<8} {'Val Loss':<10} {'Probe Acc'}")
    print("-" * 60)
    for config in CONFIGS:
        for method in methods:
            r = results.get((config, method), {})
            acc = r.get("probe_acc")
            acc_str = f"{100*acc:.1f}%" if acc is not None else "N/A"
            vl = r.get("min_val_loss", float("inf"))
            vl_str = f"{vl:.4f}" if vl != float("inf") else "N/A"
            print(f"{config:<6} {method:<14} {r.get('best_lr','?'):<8} {r.get('best_wd','?'):<8} {vl_str:<10} {acc_str}")
    print()


def plot_heatmap(results: dict, methods: List[str], out_path: Path, title: str):
    n_rows = len(CONFIGS)
    n_cols = len(methods)

    # Build matrix
    mat = np.full((n_rows, n_cols), np.nan)
    for ri, config in enumerate(CONFIGS):
        for ci, method in enumerate(methods):
            r = results.get((config, method), {})
            acc = r.get("probe_acc")
            if acc is not None:
                mat[ri, ci] = acc

    # Figure layout: add left margin for group labels
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.30, right=0.88, top=0.88, bottom=0.06)

    cmap = plt.cm.Blues
    cmap.set_bad(color="#e0e0e0")  # gray for missing

    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Probe Accuracy (3 inputs)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100*x:.0f}%"))

    # Cell text
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = mat[ri, ci]
            if np.isnan(v):
                ax.text(ci, ri, "N/A", ha="center", va="center",
                        fontsize=9, color="#888888")
            else:
                text_color = "white" if v > 0.55 else "black"
                ax.text(ci, ri, f"{100*v:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    # Axes ticks
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in methods], fontsize=11)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([CONFIG_DISPLAY[c] for c in CONFIGS], fontsize=10)

    # Horizontal group separators and group info
    group_boundaries = []  # row index AFTER which to draw a line
    group_bottom_info = []  # (bottom_row, chance) — last row of each group

    row_idx = 0
    for g_label, g_configs, chance in GROUPS:
        n = len(g_configs)
        bottom_row = row_idx + n - 1
        group_bottom_info.append((bottom_row, chance))
        row_idx += n
        if row_idx < n_rows:
            group_boundaries.append(row_idx - 0.5)

    for boundary in group_boundaries:
        ax.axhline(boundary, color="white", linewidth=2.5, zorder=3)
        ax.axhline(boundary, color="#555555", linewidth=0.8, linestyle="--", zorder=4)

    # Chance annotations inside bottom-left cell of each group
    # Groups 1,2,4 (Unique, Redundant, Pair-Red) have dark cells → white text
    # Group 3 (Synergy) has light cells → dark text
    chance_text_colors = ["white", "white", "#555555", "white"]
    for (bottom_row, chance), txt_color in zip(group_bottom_info, chance_text_colors):
        ax.text(-0.45, bottom_row + 0.38, f"chance: {100*chance:.1f}%",
                ha="left", va="bottom",
                fontsize=6.5, color=txt_color, style="italic",
                transform=ax.transData, zorder=5)

    # Title & subtitle
    ax.set_title(title, fontsize=13, fontweight="bold", pad=18)

    # Outer border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#333333")

    # Column separators
    for ci in range(1, n_cols):
        ax.axvline(ci - 0.5, color="#cccccc", linewidth=0.6, zorder=2)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading best-tuned results from LR/WD search...")
    base_results = load_best_results(BASE_METHODS)
    print_summary_table(base_results, BASE_METHODS, "Base SSL methods")
    plot_heatmap(
        base_results,
        BASE_METHODS,
        OUT_PATH_BASE,
        "Which SSL Objective Captures Which PID Atom?",
    )

    all_results = load_best_results(ALL_METHODS)
    print_summary_table(all_results, ALL_METHODS, "All methods including masking")
    plot_heatmap(
        all_results,
        ALL_METHODS,
        OUT_PATH_ALL,
        "Which SSL Objective Captures Which PID Atom? Including Masked Prediction Methods",
    )
