from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator, all_pid_names


PLOT_DIR = Path("test_outputs/pid_sar3")


def _ensure_plot_dir() -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return PLOT_DIR


def _split_train_test(X: np.ndarray, Y: np.ndarray, frac: float = 0.7) -> Tuple[np.ndarray, ...]:
    n = X.shape[0]
    n_train = max(2, int(n * frac))
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def _linear_r2(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, ridge: float = 1e-4) -> float:
    if X_test.shape[0] == 0:
        X_train, Y_train, X_test, Y_test = _split_train_test(X_train, Y_train, frac=0.5)

    Xtr = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    Xte = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)
    XtX = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1])
    W = np.linalg.solve(XtX, Xtr.T @ Y_train)
    pred = Xte @ W

    ss_res = float(np.sum((Y_test - pred) ** 2))
    y_mean = np.mean(Y_test, axis=0, keepdims=True)
    ss_tot = float(np.sum((Y_test - y_mean) ** 2)) + 1e-8
    return 1.0 - ss_res / ss_tot


def _random_feature_r2(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    n_features: int = 256,
    seed: int = 0,
    ridge: float = 1e-3,
) -> float:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / np.sqrt(max(1, X_train.shape[1])), size=(X_train.shape[1], n_features))
    b = rng.uniform(-np.pi, np.pi, size=(n_features,))
    phi_tr = np.tanh(X_train @ w + b)
    phi_te = np.tanh(X_test @ w + b)
    return _linear_r2(phi_tr, Y_train, phi_te, Y_test, ridge=ridge)


def _dependence_proxy(X: np.ndarray, Y: np.ndarray) -> float:
    Xtr, Ytr, Xte, Yte = _split_train_test(X, Y)
    r2_xy = _linear_r2(Xtr, Ytr, Xte, Yte)
    r2_yx = _linear_r2(Ytr, Xtr, Yte, Xte)
    return float(0.5 * (r2_xy + r2_yx))


def _synergy_delta(xi: np.ndarray, xj: np.ndarray, xk: np.ndarray) -> Tuple[float, float, float]:
    n = xi.shape[0]
    n_train = max(2, int(0.7 * n))
    xi_tr, xi_te = xi[:n_train], xi[n_train:]
    xj_tr, xj_te = xj[:n_train], xj[n_train:]
    xk_tr, xk_te = xk[:n_train], xk[n_train:]

    r2_i = _random_feature_r2(xi_tr, xk_tr, xi_te, xk_te, n_features=256, seed=101)
    r2_j = _random_feature_r2(xj_tr, xk_tr, xj_te, xk_te, n_features=256, seed=202)
    xij_tr = np.concatenate([xi_tr, xj_tr], axis=1)
    xij_te = np.concatenate([xi_te, xj_te], axis=1)
    r2_joint = _random_feature_r2(xij_tr, xk_tr, xij_te, xk_te, n_features=384, seed=303)
    delta = r2_joint - max(r2_i, r2_j)
    return float(delta), float(r2_joint), float(max(r2_i, r2_j))


def _make_generator(seed: int = 0, sigma: float = 0.45) -> PIDSar3DatasetGenerator:
    cfg = PIDDatasetConfig(
        d=32,
        m=8,
        sigma=sigma,
        hop_choices=(1, 2, 3, 4),
        rho_choices=(0.2, 0.5, 0.8),
        seed=seed,
        deleakage_fit_samples=1536,
    )
    return PIDSar3DatasetGenerator(cfg)


def _generate_fixed_pid(gen: PIDSar3DatasetGenerator, pid_id: int, n: int) -> Dict[str, np.ndarray]:
    return gen.generate(n=n, pid_ids=[pid_id] * n)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def test_dataset_shapes_and_metadata():
    gen = _make_generator(seed=7)
    batch = gen.generate(64)

    assert batch["x1"].shape == (64, gen.cfg.d)
    assert batch["x2"].shape == (64, gen.cfg.d)
    assert batch["x3"].shape == (64, gen.cfg.d)
    assert batch["pid_id"].shape == (64,)
    assert set(np.unique(batch["pid_id"])).issubset(set(range(10)))

    synergy_mask = batch["pid_id"] >= 7
    if np.any(synergy_mask):
        assert np.all(np.isin(batch["hop"][synergy_mask], np.array(gen.cfg.hop_choices)))

    nonsynergy_mask = batch["pid_id"] < 7
    if np.any(nonsynergy_mask):
        assert np.all(batch["hop"][nonsynergy_mask] == 0)


def test_plot_pairwise_dependence_signatures():
    out_dir = _ensure_plot_dir()
    gen = _make_generator(seed=11)
    pid_names = all_pid_names()
    pair_keys = [("x1", "x2"), ("x1", "x3"), ("x2", "x3")]
    pair_labels = ["D(1,2)", "D(1,3)", "D(2,3)"]

    scores = np.zeros((10, 3), dtype=np.float32)
    for pid_id in range(10):
        batch = _generate_fixed_pid(gen, pid_id=pid_id, n=900)
        for j, (a, b) in enumerate(pair_keys):
            scores[pid_id, j] = _dependence_proxy(batch[a], batch[b])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(scores, aspect="auto", cmap="viridis")
    ax.set_xticks(range(3))
    ax.set_xticklabels(pair_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(pid_names)
    ax.set_title("Pairwise dependence proxy on raw X (higher = stronger linear dependence)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _savefig(out_dir / "pairwise_dependence_heatmap.png")

    # Basic signature sanity checks.
    assert scores[3, 0] > scores[3, 1] and scores[3, 0] > scores[3, 2]  # R12
    assert scores[4, 1] > scores[4, 0] and scores[4, 1] > scores[4, 2]  # R13
    assert scores[5, 2] > scores[5, 0] and scores[5, 2] > scores[5, 1]  # R23
    assert np.mean(scores[6]) > 0.05  # R123 should induce broad cross-view dependence


def test_plot_redundancy_monotonicity_vs_rho():
    out_dir = _ensure_plot_dir()
    gen = _make_generator(seed=13)

    rho_values = [0.2, 0.5, 0.8]
    r12_scores: List[float] = []
    r123_scores: List[float] = []

    for rho in rho_values:
        # Exact-rho datasets by temporarily overriding choices.
        old_choices = gen.cfg.rho_choices
        gen.cfg.rho_choices = (rho,)
        batch_r12 = _generate_fixed_pid(gen, pid_id=3, n=900)
        batch_r123 = _generate_fixed_pid(gen, pid_id=6, n=900)
        gen.cfg.rho_choices = old_choices

        r12_scores.append(_dependence_proxy(batch_r12["x1"], batch_r12["x2"]))
        d12 = _dependence_proxy(batch_r123["x1"], batch_r123["x2"])
        d13 = _dependence_proxy(batch_r123["x1"], batch_r123["x3"])
        d23 = _dependence_proxy(batch_r123["x2"], batch_r123["x3"])
        r123_scores.append(float(np.mean([d12, d13, d23])))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rho_values, r12_scores, marker="o", label="R12: D(x1,x2)")
    ax.plot(rho_values, r123_scores, marker="s", label="R123: mean pairwise D")
    ax.set_xlabel("rho (redundancy overlap)")
    ax.set_ylabel("Dependence proxy")
    ax.set_title("Redundancy signatures should increase with rho")
    ax.grid(alpha=0.3)
    ax.legend()
    _savefig(out_dir / "redundancy_monotonicity_vs_rho.png")

    assert r12_scores[0] < r12_scores[-1]
    assert r123_scores[0] < r123_scores[-1]


def test_plot_synergy_delta_vs_hop():
    out_dir = _ensure_plot_dir()
    gen = _make_generator(seed=17, sigma=0.2)

    hop_values = [1, 2, 3, 4]
    synergies = {
        7: (("x1", "x2"), "x3", "S12->3"),
        8: (("x1", "x3"), "x2", "S13->2"),
        9: (("x2", "x3"), "x1", "S23->1"),
    }

    old_hops = gen.cfg.hop_choices
    curves: Dict[str, List[float]] = {v[2]: [] for v in synergies.values()}
    wrong_direction_curve: List[float] = []

    for hop in hop_values:
        gen.cfg.hop_choices = (hop,)

        wrong_vals = []
        for pid_id, ((src_a, src_b), target, label) in synergies.items():
            batch = _generate_fixed_pid(gen, pid_id=pid_id, n=1200)
            delta, _, _ = _synergy_delta(batch[src_a], batch[src_b], batch[target])
            curves[label].append(delta)

            # Wrong-direction control: predict a source from the two views not matching the generative target.
            if pid_id == 7:
                wrong_delta, _, _ = _synergy_delta(batch["x1"], batch["x3"], batch["x2"])
                wrong_vals.append(wrong_delta)
        wrong_direction_curve.append(float(np.mean(wrong_vals)))

    gen.cfg.hop_choices = old_hops

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for label, vals in curves.items():
        ax.plot(hop_values, vals, marker="o", label=label)
    ax.plot(hop_values, wrong_direction_curve, marker="x", linestyle="--", label="Wrong direction control")
    ax.set_xlabel("hop (synergy depth)")
    ax.set_ylabel("Synergy proxy Δ = R²_joint - max(single)")
    ax.set_title("Directional synergy proxy on raw X")
    ax.grid(alpha=0.3)
    ax.legend()
    _savefig(out_dir / "synergy_delta_vs_hop.png")

    # Smoke-level assertions for plotting stability; probe strength depends on noise/de-leakage.
    for vals in curves.values():
        assert np.all(np.isfinite(vals))
        assert len(vals) == len(hop_values)
    assert np.all(np.isfinite(wrong_direction_curve))


if __name__ == "__main__":
    # Convenience local runner without pytest.
    test_dataset_shapes_and_metadata()
    test_plot_pairwise_dependence_signatures()
    test_plot_redundancy_monotonicity_vs_rho()
    test_plot_synergy_delta_vs_hop()
    print(f"Saved plots to {PLOT_DIR.resolve()}")
