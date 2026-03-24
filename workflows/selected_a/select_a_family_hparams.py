from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


FAMILY_CONFIGS = {
    "unique": ["A1", "A2", "A3"],
    "redundant": ["A4", "A5", "A6", "A7"],
    "synergy": ["A8", "A9", "A10", "A11"],
    "pairred": ["A12", "A13", "A14"],
}

CONFIG_TO_FAMILY = {
    config: family
    for family, configs in FAMILY_CONFIGS.items()
    for config in configs
}

PAIRRED_QUERY = {
    "A12": "x12_to_x3",
    "A13": "x13_to_x2",
    "A14": "x23_to_x1",
}


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_probe_acc(run_dir: Path) -> float:
    probe_path = run_dir / "probe" / "probe_results.json"
    if not probe_path.exists():
        return float("nan")
    with probe_path.open() as f:
        data = json.load(f)
    try:
        return float(data["overall"]["z1_z2_z3"])
    except Exception:
        return float("nan")


def _load_pretrain_val(run_dir: Path) -> float:
    history_path = run_dir / "pretrain" / "history.csv"
    if not history_path.exists():
        return float("inf")
    best = float("inf")
    with history_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                best = min(best, float(row["val_loss"]))
            except Exception:
                pass
    return best


def _load_retrieval_metrics(run_dir: Path, config: str) -> Dict[str, float]:
    retrieval_path = run_dir / "probe" / "retrieval_summary.csv"
    best_linear = float("nan")
    pairred_linear = float("nan")
    pairred_zero = float("nan")
    if not retrieval_path.exists():
        return {
            "best_linear_r1": best_linear,
            "pairred_linear_r1": pairred_linear,
            "pairred_zero_r1": pairred_zero,
        }

    with retrieval_path.open() as f:
        rows = list(csv.DictReader(f))

    linear_vals = [_float_or_nan(row["r_at_1"]) for row in rows if row.get("mode") == "linear_adapter"]
    if linear_vals:
        best_linear = max(linear_vals)

    target_query = PAIRRED_QUERY.get(config)
    if target_query is not None:
        pairred_linear_vals = [
            _float_or_nan(row["r_at_1"])
            for row in rows
            if row.get("query") == target_query and row.get("mode") == "linear_adapter"
        ]
        pairred_zero_vals = [
            _float_or_nan(row["r_at_1"])
            for row in rows
            if row.get("query") == target_query and row.get("mode") == "zero_shot"
        ]
        if pairred_linear_vals:
            pairred_linear = float(sum(pairred_linear_vals) / len(pairred_linear_vals))
        if pairred_zero_vals:
            pairred_zero = float(sum(pairred_zero_vals) / len(pairred_zero_vals))

    return {
        "best_linear_r1": best_linear,
        "pairred_linear_r1": pairred_linear,
        "pairred_zero_r1": pairred_zero,
    }


def _mean(values: List[float]) -> float:
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _group_fields(fieldnames: List[str]) -> List[str]:
    excluded = {"config", "run_dir", "seed"}
    return [name for name in fieldnames if name not in excluded]


def _sort_key(candidate: Dict[str, object]) -> Tuple[float, float, float, float]:
    family = str(candidate["family"])
    if family == "pairred":
        return (
            float(candidate["mean_pairred_linear_r1"]),
            float(candidate["mean_pairred_zero_r1"]),
            float(candidate["mean_cls_acc"]),
            -float(candidate["mean_pretrain_val_loss"]),
        )
    return (
        float(candidate["mean_cls_acc"]),
        float(candidate["mean_best_linear_r1"]),
        -float(candidate["mean_pretrain_val_loss"]),
        0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Manifest is empty")

    for row in rows:
        config = row.get("config")
        if not config:
            raise SystemExit("Manifest rows must contain a config field")
        row.setdefault("family", CONFIG_TO_FAMILY.get(config, "unknown"))

    group_fields = _group_fields(list(rows[0].keys()))
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    candidates: List[Dict[str, object]] = []
    for key, members in grouped.items():
        configs = [row["config"] for row in members]
        metrics = []
        for row in members:
            run_dir = Path(row["run_dir"])
            config = row["config"]
            retrieval_metrics = _load_retrieval_metrics(run_dir, config)
            metrics.append(
                {
                    "cls_acc": _load_probe_acc(run_dir),
                    "pretrain_val": _load_pretrain_val(run_dir),
                    **retrieval_metrics,
                }
            )

        candidate: Dict[str, object] = {field: members[0][field] for field in group_fields}
        candidate["n_configs"] = len(set(configs))
        candidate["mean_cls_acc"] = _mean([m["cls_acc"] for m in metrics])
        candidate["mean_best_linear_r1"] = _mean([m["best_linear_r1"] for m in metrics])
        candidate["mean_pairred_linear_r1"] = _mean([m["pairred_linear_r1"] for m in metrics])
        candidate["mean_pairred_zero_r1"] = _mean([m["pairred_zero_r1"] for m in metrics])
        candidate["mean_pretrain_val_loss"] = _mean([m["pretrain_val"] for m in metrics])
        candidates.append(candidate)

    candidate_fields = list(group_fields) + [
        "n_configs",
        "mean_cls_acc",
        "mean_best_linear_r1",
        "mean_pairred_linear_r1",
        "mean_pairred_zero_r1",
        "mean_pretrain_val_loss",
    ]
    with (out_prefix.parent / (out_prefix.name + "_candidates.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidates)

    selected_rows: List[Dict[str, object]] = []
    selected_json: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    methods = sorted({str(candidate["method"]) for candidate in candidates})
    families = sorted({str(candidate["family"]) for candidate in candidates})
    for family in families:
        family_size = len(FAMILY_CONFIGS[family])
        for method in methods:
            family_candidates = [
                candidate for candidate in candidates
                if candidate["family"] == family and candidate["method"] == method
            ]
            if not family_candidates:
                continue
            max_n = max(int(candidate["n_configs"]) for candidate in family_candidates)
            filtered = [candidate for candidate in family_candidates if int(candidate["n_configs"]) == max_n]
            best = max(filtered, key=_sort_key)
            best = dict(best)
            best["selection_metric"] = "pairred_linear_r1" if family == "pairred" else "cls_acc"
            selected_rows.append(best)
            selected_json[family][method] = best

    selected_fields = candidate_fields + ["selection_metric"]
    with (out_prefix.parent / (out_prefix.name + "_best.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    with (out_prefix.parent / (out_prefix.name + "_best.json")).open("w", encoding="utf-8") as f:
        json.dump(selected_json, f, indent=2)


if __name__ == "__main__":
    main()
