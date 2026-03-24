from __future__ import annotations

import csv
import json
from copy import copy
from pathlib import Path
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AGGREGATED_ROOT = REPO_ROOT / "test_outputs" / "aggregated_results"
SUPERVISED_ROOT = REPO_ROOT / "test_outputs" / "v3_runs_A_supervised"
SUPERVISED_SYNERGY_ROOT = REPO_ROOT / "test_outputs" / "supervised_synergy_benchmark_table"
OUT_PATH = REPO_ROOT / "test_outputs" / "selected_linear_unified_heatmap.png"

CONFIGS = ["A1", "A4", "A7", "A8", "A11", "A12"]
CONFIG_DISPLAY = {
    "A1": "Unique X1",
    "A4": "Redundant X1-X2",
    "A7": "Redundant X1-X2-X3",
    "A8": "Synergy X1-X2",
    "A11": "Synergy X1-X2-X3",
    "A12": "Pair-Red 12->3",
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


def _load_supervised_score(run_dir: Path) -> float:
    best = float("nan")
    for name in ("e2e_history.csv", "e2e_history.json"):
        path = run_dir / name
        if not path.exists():
            continue
        if path.suffix == ".csv":
            with path.open() as f:
                for row in csv.DictReader(f):
                    try:
                        val = float(row["val_acc"])
                    except Exception:
                        continue
                    if np.isnan(best) or val > best:
                        best = val
        else:
            for row in _read_json(path):
                try:
                    val = float(row["val_acc"])
                except Exception:
                    continue
                if np.isnan(best) or val > best:
                    best = val
    return best


def _complete_supervised(run_dir: Path) -> bool:
    return (run_dir / "e2e_final.pt").exists() and (
        (run_dir / "e2e_history.csv").exists() or (run_dir / "e2e_history.json").exists()
    )


def _iter_supervised(config: str) -> Iterator[Path]:
    if config in {"A8", "A11"}:
        synergy_dir = SUPERVISED_SYNERGY_ROOT / config / "encoder_sub_label"
        if synergy_dir.exists():
            yield synergy_dir
    config_root = SUPERVISED_ROOT / config
    if config_root.exists():
        for seed_dir in sorted(config_root.glob("seed_*")):
            if seed_dir.is_dir():
                yield seed_dir


def _iter_ssl(config: str, method: str) -> Iterator[Path]:
    root = AGGREGATED_ROOT / config / method
    if not root.exists():
        return
    for hp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        yield hp_dir


def _complete_ssl(run_dir: Path) -> bool:
    return (run_dir / "probe_linear" / "probe_results.json").exists()


def _load_linear_score(run_dir: Path) -> float:
    path = run_dir / "probe_linear" / "probe_results.json"
    if path.exists():
        try:
            return float(_read_json(path)["overall"]["z1_z2_z3"])
        except Exception:
            pass
    return float("nan")


def _choose_best(config: str, method: str):
    best = None
    if method == "supervised":
        for run_dir in _iter_supervised(config):
            if not _complete_supervised(run_dir):
                continue
            score = _load_supervised_score(run_dir)
            if np.isnan(score):
                continue
            if best is None or score > best:
                best = score
        return best
    for run_dir in _iter_ssl(config, method):
        if not _complete_ssl(run_dir):
            continue
        score = _load_linear_score(run_dir)
        if np.isnan(score):
            continue
        if best is None or score > best:
            best = score
    return best


def _counts(config: str, method: str) -> Tuple[int, int]:
    if method == "supervised":
        runs = list(_iter_supervised(config))
        return sum(1 for r in runs if _complete_supervised(r)), len(runs)
    runs = list(_iter_ssl(config, method))
    return sum(1 for r in runs if _complete_ssl(r)), len(runs)


def plot() -> None:
    scores = np.full((len(METHODS), len(CONFIGS)), np.nan)
    for ri, method in enumerate(METHODS):
        for ci, config in enumerate(CONFIGS):
            score = _choose_best(config, method)
            if score is not None:
                scores[ri, ci] = score

    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = copy(plt.cm.Blues)
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(scores, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Best Linear Probe / Supervised Score", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100*x:.0f}%"))

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([CONFIG_DISPLAY[c] for c in CONFIGS], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHODS], fontsize=11)

    for ri in range(len(METHODS)):
        for ci in range(len(CONFIGS)):
            v = scores[ri, ci]
            text_color = "white" if not np.isnan(v) and v >= 0.45 else "black"
            center = "N/A" if np.isnan(v) else f"{100*v:.1f}%"
            ax.text(ci, ri, center, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)

    for ri in range(1, len(METHODS)):
        ax.axhline(ri - 0.5, color="#d0d0d0", linewidth=0.6, zorder=3)
    for ci in range(1, len(CONFIGS)):
        ax.axvline(ci - 0.5, color="#d0d0d0", linewidth=0.6, zorder=3)

    ax.set_title("Linear Probe Results", fontsize=14, pad=24)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved linear heatmap to: {OUT_PATH}")


if __name__ == "__main__":
    plot()
