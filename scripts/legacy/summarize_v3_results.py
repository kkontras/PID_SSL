from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from data.dataset_v3 import ALL_CONFIGS


METHOD_ORDER = ["simclr", "pairwise_nce", "triangle", "confu", "masked_raw", "masked_emb", "comm", "infmask"]
METHOD_DISPLAY = {
    "simclr": "SimCLR",
    "pairwise_nce": "Pairwise NCE",
    "triangle": "TRIANGLE",
    "confu": "ConFu",
    "masked_raw": "Masked Raw",
    "masked_emb": "Masked Emb",
    "comm": "CoMM",
    "infmask": "InfMasking",
}
CONFIG_ORDER = ["A1", "A4", "A8", "A12", "A13", "A14", "B4", "B10", "C2", "C3"]
SINGLE_ATOM_HEATMAP_ORDER = ["A1", "A4", "A8", "A12", "A13", "A14"]

DISPLAY_TITLES = {
    "A1": "Unique X1",
    "A2": "Unique X2",
    "A3": "Unique X3",
    "A4": "Redundant X1-X2",
    "A5": "Redundant X1-X3",
    "A6": "Redundant X2-X3",
    "A7": "Redundant X1-X2-X3",
    "A8": "Synergy X1-X2",
    "A9": "Synergy X1-X3",
    "A10": "Synergy X2-X3",
    "A11": "Synergy X1-X2-X3",
    "A12": "PairRed X1-X2 to X3",
    "A13": "PairRed X1-X3 to X2",
    "A14": "PairRed X2-X3 to X1",
    "B4": "Mixed Unique/Redundant/Synergy",
    "B10": "Mixed Redundant123/Synergy123",
    "C2": "Stress Synergy Trio",
    "C3": "Stress Redundant+Synergy",
}


def _config_label(config: str) -> str:
    return DISPLAY_TITLES.get(config, config)


def _load_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for probe_json in sorted(root.glob("*/**/probe/probe_results.json")):
        config, method, seed = probe_json.parts[-5], probe_json.parts[-4], probe_json.parts[-3]
        with probe_json.open() as f:
            probe = json.load(f)

        cls_acc = float(probe["overall"]["z1_z2_z3"])
        per_atom = probe.get("per_atom_all_modalities", {})
        per_atom_mean = float(np.mean(list(per_atom.values()))) if per_atom else float("nan")

        retrieval_path = probe_json.parent / "retrieval_summary.csv"
        retrieval_rows: List[Dict[str, str]] = []
        if retrieval_path.exists():
            with retrieval_path.open() as f:
                retrieval_rows = list(csv.DictReader(f))

        zero_vals = [float(r["r_at_1"]) for r in retrieval_rows if r["mode"] == "zero_shot"]
        linear_vals = [float(r["r_at_1"]) for r in retrieval_rows if r["mode"] == "linear_adapter"]
        best_zero = max(zero_vals) if zero_vals else float("nan")
        best_linear = max(linear_vals) if linear_vals else float("nan")
        mean_zero = float(np.mean(zero_vals)) if zero_vals else float("nan")
        mean_linear = float(np.mean(linear_vals)) if linear_vals else float("nan")
        mean_retrieval_chance = float(np.mean([float(r["chance_r1"]) for r in retrieval_rows])) if retrieval_rows else float("nan")
        n_classes = 7 ** len(ALL_CONFIGS[config])
        classification_chance = 1.0 / n_classes

        rows.append(
            {
                "config": config,
                "method": method,
                "seed": seed,
                "cls_acc": cls_acc,
                "per_atom_mean": per_atom_mean,
                "best_zero_r1": best_zero,
                "best_linear_r1": best_linear,
                "mean_zero_r1": mean_zero,
                "mean_linear_r1": mean_linear,
                "classification_chance": classification_chance,
                "retrieval_chance": mean_retrieval_chance,
            }
        )
    return rows


def _draw_config_grid(
    rows: List[Dict[str, object]],
    out_path: Path,
    metric_keys: List[str],
    metric_labels: List[str],
    title: str,
    subtitle_key: str,
    subtitle_prefix: str,
    cmap: str = "Blues",
) -> None:
    configs = [c for c in CONFIG_ORDER if any(r["config"] == c for r in rows)]
    n_slots = max(16, len(configs))
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    axes = axes.flatten()

    all_values = []
    for r in rows:
        for k in metric_keys:
            v = float(r[k])
            if np.isfinite(v):
                all_values.append(v)
    vmin = min(all_values) if all_values else 0.0
    vmax = max(all_values) if all_values else 1.0

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(configs):
            ax.axis("off")
            continue
        config = configs[ax_idx]
        mat = np.full((len(METHOD_ORDER), len(metric_keys)), np.nan, dtype=np.float32)
        for i, method in enumerate(METHOD_ORDER):
            rec = next((r for r in rows if r["config"] == config and r["method"] == method), None)
            if rec is None:
                continue
            for j, key in enumerate(metric_keys):
                mat[i, j] = float(rec[key])

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        rec0 = next((r for r in rows if r["config"] == config), None)
        baseline = float(rec0[subtitle_key]) if rec0 is not None else float("nan")
        title_text = _config_label(config)
        if ax_idx == 0 and np.isfinite(baseline):
            title_text += f"\n{subtitle_prefix}: {100.0 * baseline:.1f}%"
        ax.set_title(title_text, fontsize=9)
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER], fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    txt_color = "white" if mat[i, j] > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{100.0 * mat[i, j]:.1f}%", ha="center", va="center", fontsize=8, color=txt_color)

    fig.suptitle(title, fontsize=16, y=0.995)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.ax.set_ylabel("score % (darker blue = higher)")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.06, wspace=0.45, hspace=0.65)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_single_atom_allmod_heatmap(rows: List[Dict[str, object]], out_path: Path) -> None:
    configs = [c for c in SINGLE_ATOM_HEATMAP_ORDER if any(r["config"] == c for r in rows)]
    if not configs:
        return

    mat = np.full((len(METHOD_ORDER), len(configs)), np.nan, dtype=np.float32)
    for i, method in enumerate(METHOD_ORDER):
        for j, config in enumerate(configs):
            rec = next((r for r in rows if r["config"] == config and r["method"] == method), None)
            if rec is not None:
                mat[i, j] = float(rec["cls_acc"])

    finite_vals = mat[np.isfinite(mat)]
    vmin = float(np.min(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0

    fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 3.0, 5.0))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_title("Single-Atom All-Modality Classification Accuracy", fontsize=14)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([DISPLAY_TITLES.get(c, c) for c in configs], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER], fontsize=9)
    ax.set_xlabel("Single-atom config")
    ax.set_ylabel("Method")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                txt_color = "white" if mat[i, j] > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{100.0 * mat[i, j]:.1f}%", ha="center", va="center", fontsize=9, color=txt_color)

    first_config = configs[0]
    cls_chance = 1.0 / (7 ** len(ALL_CONFIGS[first_config]))
    fig.text(0.01, 0.01, f"Random Classification Accuracy: {100.0 * cls_chance:.1f}%", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_ylabel("classification accuracy % (darker blue = higher)")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_single_atom_retrieval_heatmap(rows: List[Dict[str, object]], out_path: Path) -> None:
    configs = [c for c in SINGLE_ATOM_HEATMAP_ORDER if any(r["config"] == c for r in rows)]
    if not configs:
        return

    n_rows = len(METHOD_ORDER)
    n_cols = len(configs)
    mat = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    for i, method in enumerate(METHOD_ORDER):
        for j, config in enumerate(configs):
            rec = next((r for r in rows if r["config"] == config and r["method"] == method), None)
            if rec is None:
                continue
            mat[i, j] = float(rec["best_linear_r1"])

    finite_vals = mat[np.isfinite(mat)]
    vmin = float(np.min(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.max(finite_vals)) if finite_vals.size else 1.0

    fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 3.0, 5.0))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_title("Single-Atom Retrieval: Best Linear R@1", fontsize=14)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([DISPLAY_TITLES.get(c, c) for c in configs], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER], fontsize=9)
    ax.set_xlabel("Single-atom config")
    ax.set_ylabel("Method")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                txt_color = "white" if mat[i, j] > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{100.0 * mat[i, j]:.1f}%", ha="center", va="center", fontsize=8, color=txt_color)

    first_config = configs[0]
    retrieval_chance = float(next(r["retrieval_chance"] for r in rows if r["config"] == first_config))
    fig.text(0.01, 0.01, f"Random Retrieval R@1: {100.0 * retrieval_chance:.1f}%", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_ylabel("retrieval score % (darker blue = higher)")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    root = Path("test_outputs/v3_runs")
    rows = _load_rows(root)
    if not rows:
        raise SystemExit("No probe results found under test_outputs/v3_runs")

    _draw_config_grid(
        rows,
        root / "classification_heatmaps_4x4.png",
        metric_keys=["cls_acc", "per_atom_mean"],
        metric_labels=["All-mod cls", "Per-atom mean"],
        title="V3 Classification Summary: one heatmap per config, methods x classification metrics",
        subtitle_key="classification_chance",
        subtitle_prefix="Random Classification Accuracy",
        cmap="Blues",
    )
    _draw_config_grid(
        rows,
        root / "retrieval_heatmaps_4x4.png",
        metric_keys=["best_zero_r1", "best_linear_r1", "mean_zero_r1", "mean_linear_r1"],
        metric_labels=["Best zero-shot R@1", "Best linear R@1", "Mean zero-shot R@1", "Mean linear R@1"],
        title="V3 Retrieval Summary: one heatmap per config, methods x retrieval metrics",
        subtitle_key="retrieval_chance",
        subtitle_prefix="Random Retrieval R@1",
        cmap="Blues",
    )
    _draw_single_atom_allmod_heatmap(rows, root / "single_atom_allmod_classification_heatmap.png")
    _draw_single_atom_retrieval_heatmap(rows, root / "single_atom_retrieval_heatmap.png")
    print(root / "classification_heatmaps_4x4.png")
    print(root / "retrieval_heatmaps_4x4.png")
    print(root / "single_atom_allmod_classification_heatmap.png")
    print(root / "single_atom_retrieval_heatmap.png")


if __name__ == "__main__":
    main()
