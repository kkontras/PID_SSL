from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = REPO_ROOT / "test_outputs" / "best_lrwd_nonlinear_selected"
OUT_ROOT = REPO_ROOT / "test_outputs" / "expanded_method_hparam_search_selected"
TARGET_CONFIGS = ["A8", "A11", "A12"]
METHODS = ["simclr", "pairwise_nce", "triangle", "confu", "masked_raw", "masked_emb", "comm", "infmask"]


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_base_rows() -> List[dict]:
    path = BASE_ROOT / "best_lrwd_nonlinear_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing base manifest: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def load_base_scores() -> Dict[Tuple[str, str], float]:
    scores: Dict[Tuple[str, str], float] = {}
    for row in load_base_rows():
        result_path = Path(row["nonlinear_probe_dir"]) / "nonlinear_probe_results.json"
        if not result_path.exists():
            continue
        try:
            scores[(row["config"], row["method"])] = float(_read_json(result_path)["overall"]["z1_z2_z3"])
        except Exception:
            continue
    return scores


def _grid_for_method(method: str) -> List[Dict[str, str]]:
    grids: List[Dict[str, str]] = []
    if method == "simclr":
        for tau, noise in itertools.product(
            ["0.03", "0.05", "0.07", "0.1", "0.2", "0.5", "0.8"],
            ["0.02", "0.05", "0.1", "0.2", "0.3"],
        ):
            grids.append({"tau": tau, "view_noise_std": noise})
    elif method == "pairwise_nce":
        for tau in ["0.03", "0.05", "0.07", "0.1", "0.2", "0.4", "0.8"]:
            grids.append({"tau": tau})
    elif method == "triangle":
        for tau, alpha in itertools.product(
            ["0.03", "0.05", "0.07", "0.1", "0.2", "0.4", "0.8"],
            ["0.0", "0.1", "0.25", "0.5", "1.0"],
        ):
            grids.append({"tau": tau, "triangle_alpha": alpha})
    elif method == "confu":
        for tau, fuse in itertools.product(
            ["0.03", "0.05", "0.07", "0.1", "0.2", "0.4", "0.8"],
            ["0.1", "0.25", "0.5", "0.75", "0.9"],
        ):
            grids.append({"tau": tau, "confu_fuse_weight": fuse})
    elif method == "masked_raw":
        for ratio in ["0.1", "0.2", "0.3", "0.5", "0.7", "0.85"]:
            grids.append({"mask_ratio": ratio})
    elif method == "masked_emb":
        for ratio, ema, var in itertools.product(
            ["0.1", "0.2", "0.3", "0.5", "0.7"],
            ["0.99", "0.995", "0.996", "0.999", "0.9995"],
            ["0.1", "0.25", "1.0", "4.0", "8.0"],
        ):
            grids.append({"mask_ratio": ratio, "ema_momentum": ema, "masked_emb_var_weight": var})
    elif method == "comm":
        for tau, noise in itertools.product(
            ["0.03", "0.05", "0.07", "0.1", "0.2", "0.4"],
            ["0.02", "0.05", "0.1", "0.2", "0.3"],
        ):
            grids.append({"tau": tau, "view_noise_std": noise})
    elif method == "infmask":
        for tau, noise, ratio, masks in itertools.product(
            ["0.03", "0.05", "0.07", "0.1", "0.2", "0.4"],
            ["0.02", "0.05", "0.1", "0.2"],
            ["0.1", "0.2", "0.3", "0.5", "0.7", "0.85"],
            ["1", "2", "4"],
        ):
            grids.append({"tau": tau, "view_noise_std": noise, "mask_ratio": ratio, "n_mask_samples": masks})
    return grids


def _hp_tag(method: str, hp: Dict[str, str]) -> str:
    parts = [method]
    for key in sorted(hp):
        parts.append(f"{key}_{hp[key].replace('.', 'p')}")
    return "__".join(parts)


def _train_cmd(
    *,
    python_bin: str,
    device: str,
    config: str,
    method: str,
    lr: str,
    wd: str,
    hp: Dict[str, str],
    save_dir: Path,
) -> List[str]:
    cmd = [
        python_bin,
        "train_pretrain.py",
        "--method", method,
        "--config", config,
        "--Q", "7",
        "--D", "44",
        "--D_info", "4",
        "--n_train", "12000",
        "--d_model", "64",
        "--d_z", "64",
        "--n_layers", "2",
        "--tau", hp.get("tau", "0.07"),
        "--lambda_contr", "1.0",
        "--lambda_mask", "1.0",
        "--n_mask_samples", hp.get("n_mask_samples", "1"),
        "--mask_ratio", hp.get("mask_ratio", "0.5"),
        "--batch_size", "512",
        "--epochs", "60",
        "--lr", lr,
        "--weight_decay", wd,
        "--view_noise_std", hp.get("view_noise_std", "0.1"),
        "--triangle_alpha", hp.get("triangle_alpha", "0.0"),
        "--confu_fuse_weight", hp.get("confu_fuse_weight", "0.5"),
        "--ema_momentum", hp.get("ema_momentum", "0.996"),
        "--masked_emb_var_weight", hp.get("masked_emb_var_weight", "1.0"),
        "--seed", "101",
        "--device", device,
        "--save_dir", str(save_dir),
    ]
    return cmd


def _probe_cmd(
    *,
    python_bin: str,
    device: str,
    checkpoint: Path,
    config: str,
    save_dir: Path,
) -> List[str]:
    return [
        python_bin,
        "run_nonlinear_probe.py",
        "--checkpoint", str(checkpoint),
        "--probe_config", config,
        "--Q", "7",
        "--D", "44",
        "--D_info", "4",
        "--n_probe_train", "3000",
        "--n_probe_test", "1000",
        "--probe_epochs", "300",
        "--hidden_dim", "256",
        "--device", device,
        "--save_dir", str(save_dir),
    ]


def _linear_probe_cmd(
    *,
    python_bin: str,
    device: str,
    checkpoint: Path,
    config: str,
    save_dir: Path,
) -> List[str]:
    return [
        python_bin,
        "train_probe.py",
        "--checkpoint", str(checkpoint),
        "--probe_config", config,
        "--Q", "7",
        "--D", "44",
        "--n_probe_train", "3000",
        "--n_probe_test", "1000",
        "--probe_epochs", "300",
        "--device", device,
        "--save_dir", str(save_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_bin", default="python")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_base_rows()
    base_scores = load_base_scores()
    selected = {(r["config"], r["method"]): r for r in rows}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "expanded_method_hparam_manifest.csv"
    manifest_rows: List[Dict[str, str]] = []

    for config in TARGET_CONFIGS:
        for method in METHODS:
            base = selected.get((config, method))
            if base is None:
                continue
            base_score = base_scores.get((config, method), float("nan"))
            if base_score >= args.threshold:
                print(f"skip | already >= threshold | config={config} method={method} score={base_score:.3f}")
                continue
            lr = base["lr"]
            wd = base["weight_decay"]
            print(f"expand | config={config} method={method} base_score={base_score:.3f} lr={lr} wd={wd}")
            for hp in _grid_for_method(method):
                hp_tag = _hp_tag(method, hp)
                run_dir = OUT_ROOT / config / method / hp_tag / "seed_101"
                ckpt = run_dir / "pretrain" / "final.pt"
                linear_probe_out = run_dir / "probe_linear" / "probe_results.json"
                probe_out = run_dir / "probe_nonlinear" / "nonlinear_probe_results.json"

                if args.overwrite or not ckpt.exists():
                    train_cmd = _train_cmd(
                        python_bin=args.python_bin,
                        device=args.device,
                        config=config,
                        method=method,
                        lr=lr,
                        wd=wd,
                        hp=hp,
                        save_dir=run_dir / "pretrain",
                    )
                    print("cmd:", " ".join(train_cmd[:2]), "...")
                    subprocess.run(train_cmd, cwd=REPO_ROOT, check=True)
                else:
                    print(f"skip train | existing checkpoint | {ckpt}")

                if args.overwrite or not linear_probe_out.exists():
                    linear_cmd = _linear_probe_cmd(
                        python_bin=args.python_bin,
                        device=args.device,
                        checkpoint=ckpt,
                        config=config,
                        save_dir=run_dir / "probe_linear",
                    )
                    print("cmd:", " ".join(linear_cmd[:2]), "...")
                    subprocess.run(linear_cmd, cwd=REPO_ROOT, check=True)
                else:
                    print(f"skip linear probe | existing output | {linear_probe_out}")

                if args.overwrite or not probe_out.exists():
                    probe_cmd = _probe_cmd(
                        python_bin=args.python_bin,
                        device=args.device,
                        checkpoint=ckpt,
                        config=config,
                        save_dir=run_dir / "probe_nonlinear",
                    )
                    print("cmd:", " ".join(probe_cmd[:2]), "...")
                    subprocess.run(probe_cmd, cwd=REPO_ROOT, check=True)
                else:
                    print(f"skip probe | existing output | {probe_out}")

                manifest_rows.append(
                    {
                        "config": config,
                        "method": method,
                        "base_lr": lr,
                        "base_weight_decay": wd,
                        "base_score": f"{base_score:.6f}",
                        "hp_tag": hp_tag,
                        "run_dir": str(run_dir),
                    }
                )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "method",
                "base_lr",
                "base_weight_decay",
                "base_score",
                "hp_tag",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
