from __future__ import annotations

import csv
import json
from copy import copy
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = REPO_ROOT / "test_outputs" / "best_lrwd_nonlinear_selected"
EXPANDED_ROOT = REPO_ROOT / "test_outputs" / "expanded_method_hparam_search_selected"
SUPERVISED_ROOT = REPO_ROOT / "test_outputs" / "v3_runs_A_supervised"
SUPERVISED_SYNERGY_ROOT = REPO_ROOT / "test_outputs" / "supervised_synergy_benchmark_table"
OUT_PATH = EXPANDED_ROOT / "expanded_method_hparam_selected_heatmap.png"

CONFIGS = ["A8", "A11", "A12"]
CONFIG_DISPLAY = {
    "A8": "A8\nSynergy X1-X2",
    "A11": "A11\nSynergy X1-X2-X3",
    "A12": "A12\nPair-Red 12->3",
}
METHODS = ["supervised", "simclr", "pairwise_nce", "triangle", "confu", "masked_raw", "masked_emb", "comm", "infmask"]
METHOD_DISPLAY = {
    "supervised": "Supervised",
    "simclr": "SimCLR",
    "pairwise_nce": "Pairwise NCE",
    "triangle": "TRIANGLE",
    "confu": "ConFu",
    "masked_raw": "Masked Raw",
    "masked_emb": "Masked Emb",
    "comm": "CoMM",
    "infmask": "InfMasking",
}


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _load_supervised(config: str) -> float:
    best = float("nan")
    if config in {"A8", "A11"}:
        path = SUPERVISED_SYNERGY_ROOT / config / "encoder_sub_label" / "e2e_history.csv"
        if path.exists():
            with path.open() as f:
                for row in csv.DictReader(f):
                    try:
                        val = float(row["val_acc"])
                    except Exception:
                        continue
                    if np.isnan(best) or val > best:
                        best = val
    path = SUPERVISED_ROOT / config / "seed_101" / "e2e_history.csv"
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f):
                try:
                    val = float(row["val_acc"])
                except Exception:
                    continue
                if np.isnan(best) or val > best:
                    best = val
    return best


def _load_base() -> Dict[Tuple[str, str], Tuple[float, str]]:
    rows: Dict[Tuple[str, str], Tuple[float, str]] = {}
    manifest = BASE_ROOT / "best_lrwd_nonlinear_manifest.csv"
    if not manifest.exists():
        return rows
    with manifest.open() as f:
        for row in csv.DictReader(f):
            p = Path(row["nonlinear_probe_dir"]) / "nonlinear_probe_results.json"
            if not p.exists():
                continue
            try:
                score = float(_read_json(p)["overall"]["z1_z2_z3"])
            except Exception:
                continue
            rows[(row["config"], row["method"])] = (score, f"lr={row['lr']}\nwd={row['weight_decay']}")
    return rows


def _load_expanded() -> Dict[Tuple[str, str], Tuple[float, str]]:
    best: Dict[Tuple[str, str], Tuple[float, str]] = {}
    manifest = EXPANDED_ROOT / "expanded_method_hparam_manifest.csv"
    if not manifest.exists():
        return best
    with manifest.open() as f:
        for row in csv.DictReader(f):
            p = Path(row["run_dir"]) / "probe_nonlinear" / "nonlinear_probe_results.json"
            if not p.exists():
                continue
            try:
                score = float(_read_json(p)["overall"]["z1_z2_z3"])
            except Exception:
                continue
            key = (row["config"], row["method"])
            if key not in best or score > best[key][0]:
                best[key] = (score, row["hp_tag"])
    return best


def plot() -> None:
    base = _load_base()
    expanded = _load_expanded()
    scores = np.full((len(METHODS), len(CONFIGS)), np.nan)
    labels = [["" for _ in CONFIGS] for _ in METHODS]

    for ci, config in enumerate(CONFIGS):
        scores[0, ci] = _load_supervised(config)
        labels[0][ci] = "sup"
        for ri, method in enumerate(METHODS[1:], start=1):
            key = (config, method)
            if key in base:
                scores[ri, ci], labels[ri][ci] = base[key]
            if key in expanded and (np.isnan(scores[ri, ci]) or expanded[key][0] >= scores[ri, ci]):
                scores[ri, ci], labels[ri][ci] = expanded[key]

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = copy(plt.cm.Blues)
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(scores, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Best Nonlinear Probe / Supervised Score", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100*x:.0f}%"))

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([CONFIG_DISPLAY[c] for c in CONFIGS], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHODS], fontsize=11)

    for ri in range(len(METHODS)):
        for ci in range(len(CONFIGS)):
            v = scores[ri, ci]
            color = "white" if not np.isnan(v) and v >= 0.45 else "black"
            ax.text(ci, ri - 0.08, "N/A" if np.isnan(v) else f"{100*v:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color=color)
            if labels[ri][ci]:
                ax.text(ci, ri + 0.24, labels[ri][ci][:28], ha="center", va="center", fontsize=6.5, color=color)

    for ri in range(1, len(METHODS)):
        ax.axhline(ri - 0.5, color="#d0d0d0", linewidth=0.6, zorder=3)
    for ci in range(1, len(CONFIGS)):
        ax.axvline(ci - 0.5, color="#d0d0d0", linewidth=0.6, zorder=3)

    ax.set_title(
        "Expanded Method Hyperparameter Search on A8, A11, A12\n"
        "Methods below 95% were rerun with larger method-specific grids at fixed best lr/wd.",
        fontsize=14,
        pad=24,
    )

    plt.tight_layout()
    EXPANDED_ROOT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to: {OUT_PATH}")


if __name__ == "__main__":
    plot()
