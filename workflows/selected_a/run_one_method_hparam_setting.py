from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "test_outputs" / "aggregated_results"


def hp_tag(args: argparse.Namespace) -> str:
    parts = [
        f"lr_{args.lr.replace('.', 'p')}",
        f"wd_{args.weight_decay.replace('.', 'p')}",
        f"bs_{str(args.batch_size).replace('.', 'p')}",
    ]
    if args.tau is not None:
        parts.append(f"tau_{args.tau.replace('.', 'p')}")
    if args.view_noise_std is not None:
        parts.append(f"noise_{args.view_noise_std.replace('.', 'p')}")
    if args.triangle_alpha is not None:
        parts.append(f"triangle_alpha_{args.triangle_alpha.replace('.', 'p')}")
    if args.confu_fuse_weight is not None:
        parts.append(f"cfw_{args.confu_fuse_weight.replace('.', 'p')}")
    if args.mask_ratio is not None:
        parts.append(f"mr_{args.mask_ratio.replace('.', 'p')}")
    if args.ema_momentum is not None:
        parts.append(f"ema_{args.ema_momentum.replace('.', 'p')}")
    if args.masked_emb_var_weight is not None:
        parts.append(f"var_{args.masked_emb_var_weight.replace('.', 'p')}")
    if args.n_mask_samples is not None:
        parts.append(f"kms_{args.n_mask_samples}")
    if args.use_cross_modal_tf:
        parts.append(f"cmtf_l{args.cross_modal_n_layers}_h{args.cross_modal_n_heads}_dff{args.cross_modal_d_ff}")
    return "__".join(parts)


def run(cmd: list[str]) -> None:
    print("cmd:", " ".join(cmd[:2]), "...")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--lr", required=True)
    p.add_argument("--weight_decay", required=True)
    p.add_argument("--tau", default=None)
    p.add_argument("--view_noise_std", default=None)
    p.add_argument("--triangle_alpha", default=None)
    p.add_argument("--confu_fuse_weight", default=None)
    p.add_argument("--mask_ratio", default=None)
    p.add_argument("--ema_momentum", default=None)
    p.add_argument("--masked_emb_var_weight", default=None)
    p.add_argument("--n_mask_samples", default=None)
    p.add_argument("--batch_size", default="512")
    p.add_argument("--out_root", default=str(OUT_ROOT))
    p.add_argument("--python_bin", default="python")
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    # Cross-modal Transformer
    p.add_argument("--use_cross_modal_tf", type=int, default=0)
    p.add_argument("--cross_modal_n_layers", default="2")
    p.add_argument("--cross_modal_n_heads", default="4")
    p.add_argument("--cross_modal_d_ff", default="128")
    args = p.parse_args()

    run_dir = Path(args.out_root) / args.config / args.method / hp_tag(args)
    pre_dir = run_dir / "pretraining"
    lin_dir = run_dir / "probe_linear"
    nonlin_dir = run_dir / "probe_nonlinear"
    pre_ckpt = pre_dir / "final.pt"
    lin_out = lin_dir / "probe_results.json"
    nonlin_out = nonlin_dir / "nonlinear_probe_results.json"

    if args.overwrite or not pre_ckpt.exists():
        train_cmd = [
            args.python_bin,
            "train_pretrain.py",
            "--method", args.method,
            "--config", args.config,
            "--Q", "7",
            "--D", "44",
            "--D_info", "4",
            "--n_train", "12000",
            "--d_model", "64",
            "--d_z", "64",
            "--n_layers", "2",
            "--tau", args.tau or "0.07",
            "--lambda_contr", "1.0",
            "--lambda_mask", "1.0",
            "--n_mask_samples", args.n_mask_samples or "1",
            "--mask_ratio", args.mask_ratio or "0.5",
            "--batch_size", args.batch_size,
            "--epochs", "60",
            "--lr", args.lr,
            "--weight_decay", args.weight_decay,
            "--view_noise_std", args.view_noise_std or "0.1",
            "--triangle_alpha", args.triangle_alpha or "0.0",
            "--confu_fuse_weight", args.confu_fuse_weight or "0.5",
            "--ema_momentum", args.ema_momentum or "0.996",
            "--masked_emb_var_weight", args.masked_emb_var_weight or "1.0",
            "--use_cross_modal_tf", str(args.use_cross_modal_tf),
            "--cross_modal_n_layers", args.cross_modal_n_layers,
            "--cross_modal_n_heads", args.cross_modal_n_heads,
            "--cross_modal_d_ff", args.cross_modal_d_ff,
            "--seed", "101",
            "--device", args.device,
            "--save_dir", str(pre_dir),
        ]
        run(train_cmd)

    if args.overwrite or not lin_out.exists():
        lin_cmd = [
            args.python_bin,
            "train_probe.py",
            "--checkpoint", str(pre_ckpt),
            "--probe_config", args.config,
            "--Q", "7",
            "--D", "44",
            "--n_probe_train", "3000",
            "--n_probe_test", "1000",
            "--probe_epochs", "300",
            "--device", args.device,
            "--save_dir", str(lin_dir),
        ]
        run(lin_cmd)

    if args.overwrite or not nonlin_out.exists():
        nonlin_cmd = [
            args.python_bin,
            "run_nonlinear_probe.py",
            "--checkpoint", str(pre_ckpt),
            "--probe_config", args.config,
            "--Q", "7",
            "--D", "44",
            "--D_info", "4",
            "--n_probe_train", "3000",
            "--n_probe_test", "1000",
            "--probe_epochs", "300",
            "--hidden_dim", "256",
            "--device", args.device,
            "--save_dir", str(nonlin_dir),
        ]
        run(nonlin_cmd)

    meta = {
        "config": args.config,
        "method": args.method,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "out_root": str(Path(args.out_root)),
        "hyperparameters": {
            k: getattr(args, k)
            for k in [
                "tau",
                "view_noise_std",
                "triangle_alpha",
                "confu_fuse_weight",
                "mask_ratio",
                "ema_momentum",
                "masked_emb_var_weight",
                "n_mask_samples",
            ]
            if getattr(args, k) is not None
        },
        "cross_modal_tf": {
            "use_cross_modal_tf": bool(args.use_cross_modal_tf),
            "cross_modal_n_layers": args.cross_modal_n_layers,
            "cross_modal_n_heads": args.cross_modal_n_heads,
            "cross_modal_d_ff": args.cross_modal_d_ff,
        } if args.use_cross_modal_tf else {},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"Completed run: {run_dir}")


if __name__ == "__main__":
    main()
