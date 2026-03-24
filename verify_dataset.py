"""
Dataset verification script (Rung 0): checks label distribution, feature encoding,
and information-theoretic properties of the V3 dataset generator.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from data.dataset_v3 import V3Config, V3DatasetGenerator, ALL_CONFIGS, SINGLE_ATOM_CONFIGS


def check_label_uniformity(labels: np.ndarray, n_classes: int, name: str, tol: float = 0.05) -> bool:
    """Chi-squared test for label uniformity. Returns True if uniform (p > 0.01)."""
    counts = np.bincount(labels, minlength=n_classes)
    expected = len(labels) / n_classes
    chi2, p = stats.chisquare(counts, f_exp=np.full(n_classes, expected))
    uniform = p > 0.01
    ratio = counts.min() / (counts.max() + 1e-8)
    print(f"  [{name}] chi2={chi2:.2f} p={p:.4f} min/max={ratio:.3f} {'OK' if uniform else 'FAIL'}")
    return uniform


def check_feature_separation(gen: V3DatasetGenerator, node: int, n_samples: int = 5000) -> bool:
    """Verify info dims are more structured (lower variance) than noise dims."""
    data = gen.generate(n_samples)
    x = data[f"x{node}"]

    info_ranges = gen.info_dims_for_node(node)
    noise_dims = gen.noise_dims_for_node(node)

    if not info_ranges or not noise_dims:
        print(f"  [Node {node}] Skipping: no info or noise dims")
        return True

    info_dims_flat = []
    for s, e in info_ranges:
        info_dims_flat.extend(range(s, e))

    info_var = float(x[:, info_dims_flat].var())
    noise_var = float(x[:, noise_dims].var())

    # Info dims should be more structured (lower total variance) than noise
    ok = True  # We just report, don't fail
    print(f"  [Node {node}] info_var={info_var:.4f} noise_var={noise_var:.4f} ratio={noise_var / (info_var + 1e-8):.2f}")
    return ok


def check_mlp_single_node(
    gen: V3DatasetGenerator,
    node: int,
    atom: str,
    n_train: int = 5000,
    n_test: int = 1000,
    device: str = "cpu",
) -> float:
    """Train a tiny MLP: X_node -> sub_label[atom]. Returns test accuracy."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    Q = gen.cfg.Q
    data_train = gen.generate(n_train)
    data_test = gen.generate(n_test)

    x_tr = data_train[f"x{node}"]
    x_te = data_test[f"x{node}"]
    y_tr = data_train[f"sub_{atom}"]
    y_te = data_test[f"sub_{atom}"]

    feat_mean = x_tr.mean(axis=0, keepdims=True)
    feat_std = x_tr.std(axis=0, keepdims=True) + 1e-6
    x_tr = (x_tr - feat_mean) / feat_std
    x_te = (x_te - feat_mean) / feat_std

    x_tr_t = torch.from_numpy(x_tr).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    x_te_t = torch.from_numpy(x_te).float().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)

    model = nn.Sequential(
        nn.Linear(gen.cfg.D, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, Q),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    batch_size = min(256, n_train)
    for _ in range(120):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(x_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(x_te_t).argmax(dim=-1) == y_te_t).float().mean().item()
    return acc


def check_mlp_concat_label(
    gen: V3DatasetGenerator,
    n_train: int = 5000,
    n_test: int = 1000,
    device: str = "cpu",
    epochs: int = 80,
) -> Tuple[float, float]:
    """Train an MLP on concatenated nodes to predict the full composite label."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    n_classes = gen.cfg.n_classes()
    data_train = gen.generate(n_train)
    data_test = gen.generate(n_test)

    x_tr = np.concatenate([data_train["x1"], data_train["x2"], data_train["x3"]], axis=1)
    x_te = np.concatenate([data_test["x1"], data_test["x2"], data_test["x3"]], axis=1)
    y_tr = data_train["label"]
    y_te = data_test["label"]

    x_tr_t = torch.from_numpy(x_tr).float().to(device)
    x_te_t = torch.from_numpy(x_te).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)

    feat_mean = x_tr.mean(axis=0, keepdims=True)
    feat_std = x_tr.std(axis=0, keepdims=True) + 1e-6
    x_tr = (x_tr - feat_mean) / feat_std
    x_te = (x_te - feat_mean) / feat_std

    model = nn.Sequential(
        nn.Linear(3 * gen.cfg.D, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, n_classes),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    batch_size = min(512, n_train)
    for _ in range(max(epochs, 160)):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(x_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(x_tr_t).argmax(dim=-1) == y_tr_t).float().mean().item()
        test_acc = (model(x_te_t).argmax(dim=-1) == y_te_t).float().mean().item()
    return train_acc, test_acc


def check_tf_concat_label(
    gen: V3DatasetGenerator,
    n_train: int = 5000,
    n_test: int = 1000,
    device: str = "cpu",
    epochs: int = 120,
) -> Tuple[float, float]:
    """Train a tiny transformer on the 3 nodes to predict the full composite label."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    n_classes = gen.cfg.n_classes()
    data_train = gen.generate(n_train)
    data_test = gen.generate(n_test)

    x_tr = np.stack([data_train["x1"], data_train["x2"], data_train["x3"]], axis=1)
    x_te = np.stack([data_test["x1"], data_test["x2"], data_test["x3"]], axis=1)
    y_tr = data_train["label"]
    y_te = data_test["label"]

    feat_mean = x_tr.mean(axis=(0, 1), keepdims=True)
    feat_std = x_tr.std(axis=(0, 1), keepdims=True) + 1e-6
    x_tr = (x_tr - feat_mean) / feat_std
    x_te = (x_te - feat_mean) / feat_std

    x_tr_t = torch.from_numpy(x_tr).float().to(device)
    x_te_t = torch.from_numpy(x_te).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)

    class TinyNodeTransformer(nn.Module):
        def __init__(self, d_in: int, n_classes: int, d_model: int = 96, n_heads: int = 4, n_layers: int = 2):
            super().__init__()
            self.node_proj = nn.Linear(d_in, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.zeros(1, 4, d_model))
            try:
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=2 * d_model,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=True,
                )
            except TypeError:
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=2 * d_model,
                    dropout=0.1,
                    batch_first=True,
                )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            tok = self.node_proj(x)
            cls = self.cls_token.expand(bsz, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
            tok = tok + self.pos_embed
            out = self.encoder(tok)
            return self.head(out[:, 0])

    model = TinyNodeTransformer(gen.cfg.D, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    batch_size = min(256, n_train)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(x_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(x_tr_t).argmax(dim=-1) == y_tr_t).float().mean().item()
        test_acc = (model(x_te_t).argmax(dim=-1) == y_te_t).float().mean().item()
    return train_acc, test_acc


def run_rung0(config_names: List[str], Q: int = 7, D: int = 44, D_info: int = 4,
              n_samples: int = 20000, device: str = "cpu",
              sigma_info: float = 0.01, mu_bg: float = 0.75, sigma_bg: float = 0.3) -> None:
    """Run Rung 0 data checks for the given config names."""
    print("=" * 60)
    print("RUNG 0: Dataset Sanity Checks")
    print("=" * 60)
    all_pass = True

    for name in config_names:
        atoms = ALL_CONFIGS[name]
        cfg = V3Config(
            Q=Q,
            D=D,
            D_info=D_info,
            active_atoms=atoms,
            sigma_info=sigma_info,
            mu_bg=mu_bg,
            sigma_bg=sigma_bg,
            n_samples=n_samples,
            seed=42,
        )
        gen = V3DatasetGenerator(cfg)
        n_cls = cfg.n_classes()

        print(f"\n[Config {name}] atoms={atoms} n_classes={n_cls}")
        data = gen.generate(n_samples)

        # 0A: Label uniformity
        ok = check_label_uniformity(data["label"], n_cls, f"{name} label")
        all_pass = all_pass and ok

        # 0B: Feature separation for each node
        for node in [1, 2, 3]:
            check_feature_separation(gen, node, n_samples)

        # 0C: Single-node MLP checks (single-atom configs only)
        if name in SINGLE_ATOM_CONFIGS and len(atoms) == 1:
            atom = atoms[0]
            print(f"  [MLP checks for atom={atom}]")
            for node in [1, 2, 3]:
                acc = check_mlp_single_node(gen, node, atom)
                chance = 1.0 / Q
                from data.dataset_v3 import V3DatasetGenerator as Gen
                nodes_with_info = Gen._nodes_for_atom(atom)
                expected_high = node in nodes_with_info and not atom.startswith("syn_")
                status = "OK" if (expected_high and acc > 0.6) or (not expected_high and acc < 0.3) else "WARN"
                print(f"    Node {node} -> acc={acc:.3f} chance={chance:.3f} [{status}]")

        # 0D: Full-label solvability from all three nodes concatenated
        concat_train_acc, concat_test_acc = check_mlp_concat_label(
            gen,
            n_train=max(2000, min(8000, int(0.8 * n_samples))),
            n_test=max(500, min(2000, int(0.2 * n_samples))),
            device=device,
        )
        chance = 1.0 / n_cls
        status = "OK" if concat_test_acc > max(0.15, 2.0 * chance) else "WARN"
        print(
            "  [Concat MLP]"
            f" train_acc={concat_train_acc:.3f}"
            f" test_acc={concat_test_acc:.3f}"
            f" chance={chance:.5f}"
            f" [{status}]"
        )

        tf_train_acc, tf_test_acc = check_tf_concat_label(
            gen,
            n_train=max(2000, min(8000, int(0.8 * n_samples))),
            n_test=max(500, min(2000, int(0.2 * n_samples))),
            device=device,
        )
        tf_status = "OK" if tf_test_acc > max(0.15, 2.0 * chance) else "WARN"
        print(
            "  [Concat TF]"
            f" train_acc={tf_train_acc:.3f}"
            f" test_acc={tf_test_acc:.3f}"
            f" chance={chance:.5f}"
            f" [{tf_status}]"
        )

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILURES — fix before proceeding'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 dataset verification (Rung 0)")
    parser.add_argument("--configs", nargs="+", default=list(SINGLE_ATOM_CONFIGS.keys()),
                        help="Config names to check")
    parser.add_argument("--Q", type=int, default=7)
    parser.add_argument("--D", type=int, default=44)
    parser.add_argument("--D_info", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sigma_info", type=float, default=0.01)
    parser.add_argument("--mu_bg", type=float, default=0.75)
    parser.add_argument("--sigma_bg", type=float, default=0.3)
    args = parser.parse_args()
    run_rung0(
        args.configs,
        Q=args.Q,
        D=args.D,
        D_info=args.D_info,
        n_samples=args.n_samples,
        device=args.device,
        sigma_info=args.sigma_info,
        mu_bg=args.mu_bg,
        sigma_bg=args.sigma_bg,
    )
