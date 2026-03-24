from __future__ import annotations

import csv
import json
from copy import copy
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).parent
FAMILY_ROOT = REPO_ROOT / "test_outputs" / "v3_runs_A_family_tuning"
LEGACY_ROOT = REPO_ROOT / "test_outputs" / "v3_runs_A_lrwd_search"
SELECTED_ROOT = REPO_ROOT / "test_outputs" / "v3_runs_selected_search"
SUPERVISED_ROOT = REPO_ROOT / "test_outputs" / "v3_runs_A_supervised"
SUPERVISED_SYNERGY_ROOT = REPO_ROOT / "test_outputs" / "supervised_synergy_benchmark_table"
OUT_PATH = REPO_ROOT / "test_outputs" / "unified_A_results_heatmap.png"

CONFIGS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14"]
CONFIG_DISPLAY = {
    "A1": "A1\nUnique X1",
    "A2": "A2\nUnique X2",
    "A3": "A3\nUnique X3",
    "A4": "A4\nRedundant X1-X2",
    "A5": "A5\nRedundant X1-X3",
    "A6": "A6\nRedundant X2-X3",
    "A7": "A7\nRedundant X1-X2-X3",
    "A8": "A8\nSynergy X1-X2",
    "A9": "A9\nSynergy X1-X3",
    "A10": "A10\nSynergy X2-X3",
    "A11": "A11\nSynergy X1-X2-X3",
    "A12": "A12\nPair-Red 12->3",
    "A13": "A13\nPair-Red 13->2",
    "A14": "A14\nPair-Red 23->1",
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
PAIRRED_QUERY = {
    "A12": "x12_to_x3",
    "A13": "x13_to_x2",
    "A14": "x23_to_x1",
}
SELECTED_CONFIGS = {"A1", "A4", "A7", "A8", "A11", "A12"}


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
            rows = _read_json(path)
            for row in rows:
                try:
                    val = float(row["val_acc"])
                except Exception:
                    continue
                if np.isnan(best) or val > best:
                    best = val
    return best


def _load_cls_acc(path: Path) -> float:
    try:
        data = _read_json(path)
        return float(data["overall"]["z1_z2_z3"])
    except Exception:
        return float("nan")


def _load_pairred_r1(path: Path, config: str) -> float:
    target_query = PAIRRED_QUERY.get(config)
    if not path.exists() or target_query is None:
        return float("nan")
    vals: List[float] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("query") == target_query and row.get("mode") == "linear_adapter":
                try:
                    vals.append(float(row["r_at_1"]))
                except Exception:
                    pass
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _score_ssl_run(run_dir: Path, config: str) -> float:
    if (run_dir / "probe" / "probe_results.json").exists():
        cls_path = run_dir / "probe" / "probe_results.json"
        retr_path = run_dir / "probe" / "retrieval_summary.csv"
    elif (run_dir / "probe_linear" / "probe_results.json").exists():
        cls_path = run_dir / "probe_linear" / "probe_results.json"
        retr_path = run_dir / "probe_linear" / "retrieval_summary.csv"
    else:
        return float("nan")
    if config in PAIRRED_QUERY:
        return _load_pairred_r1(retr_path, config)
    return _load_cls_acc(cls_path)


def _complete_ssl_run(run_dir: Path) -> bool:
    return (run_dir / "pretrain" / "final.pt").exists() and (
        (run_dir / "probe" / "probe_results.json").exists()
        or (run_dir / "probe_linear" / "probe_results.json").exists()
    )


def _complete_supervised_run(run_dir: Path) -> bool:
    return (run_dir / "e2e_final.pt").exists() and (
        (run_dir / "e2e_history.csv").exists() or (run_dir / "e2e_history.json").exists()
    )


def _iter_supervised_runs(config: str) -> Iterator[Path]:
    if config in {"A8", "A9", "A10", "A11"}:
        synergy_dir = SUPERVISED_SYNERGY_ROOT / config / "encoder_sub_label"
        if synergy_dir.exists():
            yield synergy_dir
    config_root = SUPERVISED_ROOT / config
    if config_root.exists():
        for seed_dir in sorted(config_root.glob("seed_*")):
            if seed_dir.is_dir():
                yield seed_dir


def _iter_ssl_runs(config: str, method: str) -> Iterator[Path]:
    roots = [
        FAMILY_ROOT / "method" / config / method,
        FAMILY_ROOT / "optimizer" / config / method,
        LEGACY_ROOT / config / method,
    ]
    if config in SELECTED_CONFIGS:
        roots.append(SELECTED_ROOT / config / method)

    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for hp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in hp_dir.glob("seed_*") if p.is_dir()):
                resolved = seed_dir.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield seed_dir


def _choose_best(config: str, method: str) -> Optional[Tuple[float, Path]]:
    best: Optional[Tuple[float, Path]] = None
    if method == "supervised":
        for run_dir in _iter_supervised_runs(config):
            if not _complete_supervised_run(run_dir):
                continue
            score = _load_supervised_score(run_dir)
            if np.isnan(score):
                continue
            if best is None or score > best[0]:
                best = (score, run_dir)
        return best

    for run_dir in _iter_ssl_runs(config, method):
        if not _complete_ssl_run(run_dir):
            continue
        score = _score_ssl_run(run_dir, config)
        if np.isnan(score):
            continue
        if best is None or score > best[0]:
            best = (score, run_dir)
    return best


def _counts(config: str, method: str) -> Tuple[int, int]:
    total = 0
    complete = 0
    if method == "supervised":
        runs = list(_iter_supervised_runs(config))
        total = len(runs)
        complete = sum(1 for r in runs if _complete_supervised_run(r))
        return complete, total
    runs = list(_iter_ssl_runs(config, method))
    total = len(runs)
    complete = sum(1 for r in runs if _complete_ssl_run(r))
    return complete, total


def build_plot_data():
    scores = np.full((len(METHODS), len(CONFIGS)), np.nan, dtype=float)
    counts = [["0/0" for _ in CONFIGS] for _ in METHODS]
    for ri, method in enumerate(METHODS):
        for ci, config in enumerate(CONFIGS):
            complete, total = _counts(config, method)
            counts[ri][ci] = f"{complete}/{total}"
            result = _choose_best(config, method)
            if result is not None:
                scores[ri, ci] = result[0]
    return scores, counts


def plot() -> None:
    scores, counts = build_plot_data()

    fig, ax = plt.subplots(figsize=(22, 9))
    cmap = copy(plt.cm.Blues)
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(scores, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Best Completed Score", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100*x:.0f}%"))

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([CONFIG_DISPLAY[c] for c in CONFIGS], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHODS], fontsize=11)

    for ri in range(len(METHODS)):
        for ci in range(len(CONFIGS)):
            v = scores[ri, ci]
            text_color = "white" if not np.isnan(v) and v >= 0.45 else "black"
            center_label = "N/A" if np.isnan(v) else f"{100 * v:.1f}%"
            ax.text(ci, ri, center_label, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)
            ax.text(ci + 0.44, ri + 0.38, counts[ri][ci], ha="right", va="bottom", fontsize=7, color=text_color)

    for ri in range(1, len(METHODS)):
        ax.axhline(ri - 0.5, color="#d0d0d0", linewidth=0.6, zorder=3)
    for ci in range(1, len(CONFIGS)):
        ax.axvline(ci - 0.5, color="#d0d0d0", linewidth=0.6, zorder=3)
    for boundary in [2.5, 6.5, 10.5]:
        ax.axvline(boundary, color="#666666", linewidth=1.2, linestyle="--", zorder=4)

    title = "Unified A-Scenario Results (A1-A14)"
    subtitle = (
        "Aggregated across v3_runs_A_family_tuning, v3_runs_A_lrwd_search, "
        "v3_runs_selected_search, and supervised roots. "
        "Center: best completed score. Bottom-right: completed/total runs found."
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=14, pad=28)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved unified heatmap to: {OUT_PATH}")


if __name__ == "__main__":
    plot()
