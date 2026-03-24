from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AGGREGATED_ROOT = REPO_ROOT / "test_outputs" / "aggregated_results"
SUPERVISED_ROOT = REPO_ROOT / "test_outputs" / "v3_runs_A_supervised"
SUPERVISED_SYNERGY_ROOT = REPO_ROOT / "test_outputs" / "supervised_synergy_benchmark_table"
BEST_LRWD_MANIFEST = REPO_ROOT / "test_outputs" / "best_lrwd_nonlinear_selected" / "best_lrwd_nonlinear_manifest.csv"

LINEAR_CSV = REPO_ROOT / "test_outputs" / "selected_linear_unified_heatmap_best_runs.csv"
NONLINEAR_CSV = REPO_ROOT / "test_outputs" / "selected_nonlinear_unified_heatmap_best_runs.csv"

CONFIGS = ["A1", "A4", "A7", "A8", "A11", "A12"]
METHODS = ["supervised", "simclr", "pairwise_nce", "triangle", "confu", "masked_raw", "masked_emb", "comm", "infmask"]


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


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


def _complete_linear(run_dir: Path) -> bool:
    return (run_dir / "probe_linear" / "probe_results.json").exists()


def _complete_nonlinear(run_dir: Path) -> bool:
    return (run_dir / "probe_nonlinear" / "nonlinear_probe_results.json").exists()


def _load_linear_score(run_dir: Path) -> float:
    try:
        return float(_read_json(run_dir / "probe_linear" / "probe_results.json")["overall"]["z1_z2_z3"])
    except Exception:
        return float("nan")


def _load_nonlinear_score(run_dir: Path) -> float:
    try:
        return float(_read_json(run_dir / "probe_nonlinear" / "nonlinear_probe_results.json")["overall"]["z1_z2_z3"])
    except Exception:
        return float("nan")


def _load_base_lrwd() -> Dict[Tuple[str, str], Tuple[str, str]]:
    out: Dict[Tuple[str, str], Tuple[str, str]] = {}
    if not BEST_LRWD_MANIFEST.exists():
        return out
    with BEST_LRWD_MANIFEST.open() as f:
        for row in csv.DictReader(f):
            out[(row["config"], row["method"])] = (row["lr"], row["weight_decay"])
    return out


def _infer_from_hp_config(hp_config: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in hp_config.split("__"):
        if "_" not in part:
            continue
        key, value = part.split("_", 1)
        out[key] = value.replace("p", ".")
    return out


def _extract_run_info(config: str, method: str, run_dir: Path, base_lrwd: Dict[Tuple[str, str], Tuple[str, str]]) -> Dict[str, str]:
    info = {
        "config": config,
        "method": method,
        "run_dir": str(run_dir),
        "source_run_dir": "",
        "lr": "",
        "weight_decay": "",
        "tau": "",
        "view_noise_std": "",
        "triangle_alpha": "",
        "confu_fuse_weight": "",
        "mask_ratio": "",
        "ema_momentum": "",
        "masked_emb_var_weight": "",
        "n_mask_samples": "",
        "hp_config": run_dir.name,
    }
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = _read_json(meta_path)
        info["hp_config"] = str(meta.get("hp_config", run_dir.name))
        if meta.get("lr"):
            info["lr"] = str(meta["lr"])
        if meta.get("weight_decay"):
            info["weight_decay"] = str(meta["weight_decay"])
        if isinstance(meta.get("hyperparameters"), dict):
            for key in [
                "tau",
                "view_noise_std",
                "triangle_alpha",
                "confu_fuse_weight",
                "mask_ratio",
                "ema_momentum",
                "masked_emb_var_weight",
                "n_mask_samples",
            ]:
                if meta["hyperparameters"].get(key) is not None:
                    info[key] = str(meta["hyperparameters"][key])
        sources = meta.get("sources")
        if isinstance(sources, list) and sources:
            src = sources[0]
            info["source_run_dir"] = str(src.get("source_run_dir", ""))
    inferred = _infer_from_hp_config(info["hp_config"])
    info["lr"] = info["lr"] or inferred.get("lr", "")
    info["weight_decay"] = info["weight_decay"] or inferred.get("wd", "")
    info["tau"] = info["tau"] or inferred.get("tau", "")
    info["view_noise_std"] = info["view_noise_std"] or inferred.get("noise", "")
    info["triangle_alpha"] = info["triangle_alpha"] or inferred.get("triangle_alpha", "")
    info["confu_fuse_weight"] = info["confu_fuse_weight"] or inferred.get("cfw", "")
    info["mask_ratio"] = info["mask_ratio"] or inferred.get("mr", "")
    info["ema_momentum"] = info["ema_momentum"] or inferred.get("ema", "")
    info["masked_emb_var_weight"] = info["masked_emb_var_weight"] or inferred.get("var", "")
    info["n_mask_samples"] = info["n_mask_samples"] or inferred.get("kms", "")
    if (not info["lr"] or not info["weight_decay"]) and (config, method) in base_lrwd:
        info["lr"], info["weight_decay"] = base_lrwd[(config, method)]
    return info


def _best_supervised(config: str) -> Tuple[Optional[float], Optional[Path]]:
    best_score = None
    best_dir = None
    for run_dir in _iter_supervised(config):
        if not _complete_supervised(run_dir):
            continue
        score = _load_supervised_score(run_dir)
        if np.isnan(score):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_dir = run_dir
    return best_score, best_dir


def _best_ssl(config: str, method: str, nonlinear: bool) -> Tuple[Optional[float], Optional[Path]]:
    best_score = None
    best_dir = None
    for run_dir in _iter_ssl(config, method):
        if nonlinear:
            if not _complete_nonlinear(run_dir):
                continue
            score = _load_nonlinear_score(run_dir)
        else:
            if not _complete_linear(run_dir):
                continue
            score = _load_linear_score(run_dir)
        if np.isnan(score):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_dir = run_dir
    return best_score, best_dir


def _write_csv(out_path: Path, nonlinear: bool) -> None:
    base_lrwd = _load_base_lrwd()
    rows = []
    for config in CONFIGS:
        for method in METHODS:
            if method == "supervised":
                score, run_dir = _best_supervised(config)
                row = {
                    "config": config,
                    "method": method,
                    "score": "" if score is None else f"{score:.6f}",
                    "criterion": "supervised_val_acc",
                    "run_dir": "" if run_dir is None else str(run_dir),
                    "source_run_dir": "",
                    "lr": "",
                    "weight_decay": "",
                    "tau": "",
                    "view_noise_std": "",
                    "triangle_alpha": "",
                    "confu_fuse_weight": "",
                    "mask_ratio": "",
                    "ema_momentum": "",
                    "masked_emb_var_weight": "",
                    "n_mask_samples": "",
                    "hp_config": "",
                }
            else:
                score, run_dir = _best_ssl(config, method, nonlinear=nonlinear)
                if run_dir is None:
                    row = {
                        "config": config,
                        "method": method,
                        "score": "",
                        "criterion": "best_nonlinear_probe" if nonlinear else "best_linear_probe",
                        "run_dir": "",
                        "source_run_dir": "",
                        "lr": "",
                        "weight_decay": "",
                        "tau": "",
                        "view_noise_std": "",
                        "triangle_alpha": "",
                        "confu_fuse_weight": "",
                        "mask_ratio": "",
                        "ema_momentum": "",
                        "masked_emb_var_weight": "",
                        "n_mask_samples": "",
                        "hp_config": "",
                    }
                else:
                    info = _extract_run_info(config, method, run_dir, base_lrwd)
                    row = {
                        "config": config,
                        "method": method,
                        "score": f"{score:.6f}",
                        "criterion": "best_nonlinear_probe" if nonlinear else "best_linear_probe",
                        **info,
                    }
            rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "method",
                "score",
                "criterion",
                "run_dir",
                "source_run_dir",
                "lr",
                "weight_decay",
                "tau",
                "view_noise_std",
                "triangle_alpha",
                "confu_fuse_weight",
                "mask_ratio",
                "ema_momentum",
                "masked_emb_var_weight",
                "n_mask_samples",
                "hp_config",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to: {out_path}")


def main() -> None:
    _write_csv(LINEAR_CSV, nonlinear=False)
    _write_csv(NONLINEAR_CSV, nonlinear=True)


if __name__ == "__main__":
    main()
