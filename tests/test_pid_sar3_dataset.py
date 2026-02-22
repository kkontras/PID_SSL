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


def _pc1_scores(X: np.ndarray) -> np.ndarray:
    Xc = X - np.mean(X, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[0]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _mean_view_norm(batch: Dict[str, np.ndarray], key: str) -> float:
    return float(np.mean(np.linalg.norm(batch[key], axis=1)))


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


def test_plot_ur_compact_signature_grid_over_sigma():
    out_dir = _ensure_plot_dir()
    pid_ids = [0, 1, 2, 3, 4, 5, 6]
    pid_labels = ["U1", "U2", "U3", "R12", "R13", "R23", "R123"]
    pair_keys = [("x1", "x2"), ("x1", "x3"), ("x2", "x3")]
    pair_labels = ["D(1,2)", "D(1,3)", "D(2,3)"]
    sigma_values = [0.15, 0.45, 0.9]

    panel_scores: List[np.ndarray] = []
    for col, sigma in enumerate(sigma_values):
        gen = _make_generator(seed=101 + col, sigma=sigma)
        scores = np.zeros((len(pid_ids), len(pair_keys)), dtype=np.float32)
        for i, pid_id in enumerate(pid_ids):
            batch = _generate_fixed_pid(gen, pid_id=pid_id, n=700)
            for j, (a, b) in enumerate(pair_keys):
                scores[i, j] = _dependence_proxy(batch[a], batch[b])
        panel_scores.append(scores)

    vmin = min(float(np.min(s)) for s in panel_scores)
    vmax = max(float(np.max(s)) for s in panel_scores)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    im = None
    for ax, sigma, scores in zip(axes, sigma_values, panel_scores):
        im = ax.imshow(scores, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"sigma={sigma}")
        ax.set_xticks(range(3))
        ax.set_xticklabels(pair_labels, rotation=20)
        ax.set_yticks(range(len(pid_labels)))
        ax.set_yticklabels(pid_labels)
        for r in range(scores.shape[0]):
            c = int(np.argmax(scores[r]))
            ax.text(c, r, "●", ha="center", va="center", color="white", fontsize=8)

    fig.suptitle("U/R-only signatures across noise: unique stays weak, redundancy activates matching pairs")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    _savefig(out_dir / "ur_compact_signature_grid_over_sigma.png")

    low_sigma_scores = panel_scores[0]
    assert low_sigma_scores[3, 0] > low_sigma_scores[3, 1] and low_sigma_scores[3, 0] > low_sigma_scores[3, 2]
    assert low_sigma_scores[4, 1] > low_sigma_scores[4, 0] and low_sigma_scores[4, 1] > low_sigma_scores[4, 2]
    assert low_sigma_scores[5, 2] > low_sigma_scores[5, 0] and low_sigma_scores[5, 2] > low_sigma_scores[5, 1]
    assert float(np.max(low_sigma_scores[:3])) < float(np.max(low_sigma_scores[3:]))


def test_plot_ur_hyperparameter_sweeps_compact():
    out_dir = _ensure_plot_dir()
    rho_values = [0.2, 0.5, 0.8]
    sigma_values = [0.15, 0.45, 0.9]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for sigma in sigma_values:
        gen = _make_generator(seed=200 + int(100 * sigma), sigma=sigma)
        curve = []
        for rho in rho_values:
            old_choices = gen.cfg.rho_choices
            gen.cfg.rho_choices = (rho,)
            batch = _generate_fixed_pid(gen, pid_id=3, n=700)
            gen.cfg.rho_choices = old_choices
            curve.append(_dependence_proxy(batch["x1"], batch["x2"]))
        axes[0].plot(rho_values, curve, marker="o", label=f"sigma={sigma}")

    axes[0].set_title("R12 dependence vs rho")
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("D(x1, x2)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    alpha_ranges = [(0.4, 0.6), (0.8, 1.2), (1.4, 1.8)]
    xlabels = ["0.5", "1.0", "1.6"]
    x = np.arange(len(alpha_ranges), dtype=float)

    for sigma in sigma_values:
        gen = _make_generator(seed=300 + int(100 * sigma), sigma=sigma)
        u1_norms = []
        r123_norms = []
        old_alpha_min, old_alpha_max = gen.cfg.alpha_min, gen.cfg.alpha_max
        for a_min, a_max in alpha_ranges:
            gen.cfg.alpha_min, gen.cfg.alpha_max = a_min, a_max
            b_u1 = _generate_fixed_pid(gen, pid_id=0, n=500)
            b_r123 = _generate_fixed_pid(gen, pid_id=6, n=500)
            u1_norms.append(_mean_view_norm(b_u1, "x1"))
            r123_norms.append(_mean_view_norm(b_r123, "x1"))
        gen.cfg.alpha_min, gen.cfg.alpha_max = old_alpha_min, old_alpha_max

        axes[1].plot(x, u1_norms, marker="o", linestyle="-", label=f"U1 | sigma={sigma}")
        axes[1].plot(x, r123_norms, marker="s", linestyle="--", label=f"R123 | sigma={sigma}")

    axes[1].set_title("View norm vs alpha (and noise)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels)
    axes[1].set_xlabel("alpha center")
    axes[1].set_ylabel("mean ||x1||")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    _savefig(out_dir / "ur_hyperparameter_sweeps_compact.png")

    gen_chk = _make_generator(seed=222, sigma=0.15)
    chk = []
    for rho in rho_values:
        old_choices = gen_chk.cfg.rho_choices
        gen_chk.cfg.rho_choices = (rho,)
        batch = _generate_fixed_pid(gen_chk, pid_id=3, n=600)
        gen_chk.cfg.rho_choices = old_choices
        chk.append(_dependence_proxy(batch["x1"], batch["x2"]))
    assert chk[0] < chk[-1]


def test_plot_ur_intuition_scatter_examples():
    out_dir = _ensure_plot_dir()
    sigma_values = [0.15, 0.9]
    cases = [(0, "U1"), (3, "R12"), (6, "R123")]

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    corr_table = np.zeros((len(sigma_values), len(cases)), dtype=np.float32)

    for r, sigma in enumerate(sigma_values):
        gen = _make_generator(seed=400 + r, sigma=sigma)
        for c, (pid_id, label) in enumerate(cases):
            batch = _generate_fixed_pid(gen, pid_id=pid_id, n=450)
            s1 = _pc1_scores(batch["x1"])
            s2 = _pc1_scores(batch["x2"])
            corr = _safe_corr(s1, s2)
            corr_table[r, c] = corr

            ax = axes[r, c]
            ax.scatter(s1, s2, s=8, alpha=0.45, c=batch["alpha"], cmap="viridis")
            ax.set_title(f"{label} | sigma={sigma}\nr={corr:.2f}")
            ax.set_xlabel("PC1(x1)")
            ax.set_ylabel("PC1(x2)")
            ax.grid(alpha=0.2)
            lim = np.percentile(np.abs(np.concatenate([s1, s2])), 98) + 1e-6
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

    fig.suptitle("U/R intuition: pairwise structure emerges only for redundancy")
    _savefig(out_dir / "ur_intuition_scatter_examples.png")

    assert abs(float(corr_table[0, 1])) > abs(float(corr_table[0, 0]))
    assert abs(float(corr_table[0, 2])) > abs(float(corr_table[0, 0]))
    assert np.all(np.isfinite(corr_table))


def test_plot_r123_pca_all_pairs():
    """
    Companion PCA figure for R123 across all view pairs.
    This helps show that a weak PC1(x1)-PC1(x2) panel at high noise does not imply no redundancy.
    """
    out_dir = _ensure_plot_dir()
    sigma_values = [0.15, 0.9]
    pair_defs = [("x1", "x2", "1-2"), ("x1", "x3", "1-3"), ("x2", "x3", "2-3")]

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6))
    corr_table = np.zeros((len(sigma_values), len(pair_defs)), dtype=np.float32)

    for r, sigma in enumerate(sigma_values):
        gen = _make_generator(seed=480 + r, sigma=sigma)
        batch = _generate_fixed_pid(gen, pid_id=6, n=500)  # R123
        pc = {k: _pc1_scores(batch[k]) for k in ("x1", "x2", "x3")}

        for c, (a, b, label) in enumerate(pair_defs):
            s1, s2 = pc[a], pc[b]
            corr = _safe_corr(s1, s2)
            corr_table[r, c] = corr

            ax = axes[r, c]
            ax.scatter(s1, s2, s=8, alpha=0.45, c=batch["alpha"], cmap="viridis")
            ax.set_title(f"R123 | pair {label} | sigma={sigma}\nr={corr:.2f}")
            ax.set_xlabel(f"PC1({a})")
            ax.set_ylabel(f"PC1({b})")
            ax.grid(alpha=0.2)
            lim = np.percentile(np.abs(np.concatenate([s1, s2])), 98) + 1e-6
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

    fig.suptitle("R123 PCA companion: inspect all pairs, not only (x1,x2)")
    _savefig(out_dir / "r123_pca_all_pairs.png")

    # At low noise, at least one pair should clearly show alignment.
    assert float(np.max(np.abs(corr_table[0]))) > 0.1
    assert np.all(np.isfinite(corr_table))


def test_plot_atom_gain_controls_ur():
    """
    Demonstrate controllable atom-family scaling.
    We compare baseline vs boosted-U vs boosted-R vs per-pid overrides.
    """
    out_dir = _ensure_plot_dir()
    scenarios = [
        ("baseline", dict(unique_gain=1.0, redundancy_gain=1.0)),
        ("boost_U", dict(unique_gain=1.8, redundancy_gain=1.0)),
        ("boost_R", dict(unique_gain=1.0, redundancy_gain=1.8)),
        ("unequal_R", dict(unique_gain=1.0, redundancy_gain=1.0, pid_gain_overrides={3: 2.0, 6: 0.7})),
    ]
    target_pids = [(0, "U1"), (3, "R12"), (6, "R123")]

    mean_norms = np.zeros((len(scenarios), len(target_pids)), dtype=np.float32)
    d12_vals = np.zeros((len(scenarios), len(target_pids)), dtype=np.float32)

    for s_idx, (_, gains) in enumerate(scenarios):
        cfg = PIDDatasetConfig(
            d=32,
            m=8,
            sigma=0.45,
            rho_choices=(0.5,),  # fix rho to isolate gain effect
            hop_choices=(1, 2, 3, 4),
            seed=700 + s_idx,
            deleakage_fit_samples=1024,
            **gains,
        )
        gen = PIDSar3DatasetGenerator(cfg)
        for p_idx, (pid_id, _) in enumerate(target_pids):
            batch = _generate_fixed_pid(gen, pid_id=pid_id, n=700)
            mean_norms[s_idx, p_idx] = _mean_view_norm(batch, "x1")
            d12_vals[s_idx, p_idx] = _dependence_proxy(batch["x1"], batch["x2"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    x = np.arange(len(target_pids))
    width = 0.18
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    for s_idx, (label, _) in enumerate(scenarios):
        offset = (s_idx - 1.5) * width
        axes[0].bar(x + offset, mean_norms[s_idx], width=width, label=label, color=colors[s_idx], alpha=0.85)
        axes[1].bar(x + offset, d12_vals[s_idx], width=width, label=label, color=colors[s_idx], alpha=0.85)

    tick_labels = [label for _, label in target_pids]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_title("Effect of atom gain controls on observed norm (view 1)")
    axes[0].set_ylabel("mean ||x1||")
    axes[1].set_title("Effect of atom gain controls on D(1,2)")
    axes[1].set_ylabel("dependence proxy")
    axes[1].legend(fontsize=8)

    _savefig(out_dir / "atom_gain_controls_ur.png")

    # Expected directional effects (with fixed rho and same sigma).
    assert mean_norms[1, 0] > mean_norms[0, 0]  # boost_U increases U1 scale
    assert d12_vals[2, 1] > d12_vals[0, 1]      # boost_R increases R12 dependence
    assert d12_vals[3, 1] > d12_vals[0, 1]      # per-pid R12 override increases D(1,2)


def test_plot_pid_metadata_distributions():
    """
    Distribution-oriented summary across all pid atoms:
    - class counts (balanced schedule)
    - alpha distributions per pid
    - rho usage for redundancy atoms
    - hop usage for synergy atoms
    """
    out_dir = _ensure_plot_dir()
    gen = _make_generator(seed=501, sigma=0.45)
    pid_names = all_pid_names()

    n_per_pid = 350
    pid_schedule = np.repeat(np.arange(10), n_per_pid)
    batch = gen.generate(n=len(pid_schedule), pid_ids=pid_schedule.tolist())

    pid_arr = batch["pid_id"]
    alpha_arr = batch["alpha"]
    rho_arr = batch["rho"]
    hop_arr = batch["hop"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    # (1) Class counts
    counts = np.bincount(pid_arr, minlength=10)
    axes[0, 0].bar(np.arange(10), counts, color="#3b7ddd")
    axes[0, 0].set_title("Class counts (balanced generation schedule)")
    axes[0, 0].set_xlabel("pid_id")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_xticks(np.arange(10))
    axes[0, 0].set_xticklabels([str(i) for i in range(10)], rotation=0)
    axes[0, 0].grid(axis="y", alpha=0.25)

    # (2) Alpha distributions by pid (compact boxplots)
    alpha_groups = [alpha_arr[pid_arr == i] for i in range(10)]
    bp = axes[0, 1].boxplot(alpha_groups, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#70ad47")
        patch.set_alpha(0.6)
    axes[0, 1].set_title("Per-pid alpha distribution")
    axes[0, 1].set_xlabel("pid_id")
    axes[0, 1].set_ylabel("alpha")
    axes[0, 1].set_xticks(np.arange(1, 11))
    axes[0, 1].set_xticklabels([str(i) for i in range(10)])
    axes[0, 1].grid(axis="y", alpha=0.25)

    # (3) Rho values only for redundancy atoms
    red_pid_ids = [3, 4, 5, 6]
    red_labels = [pid_names[i] for i in red_pid_ids]
    red_rho_groups = [rho_arr[pid_arr == i] for i in red_pid_ids]
    axes[1, 0].hist(
        red_rho_groups,
        bins=np.array([0.05, 0.35, 0.65, 0.95]),
        label=red_labels,
        alpha=0.7,
        stacked=False,
    )
    axes[1, 0].set_title("rho distribution for redundancy atoms")
    axes[1, 0].set_xlabel("rho")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)

    # (4) Hop values only for synergy atoms
    syn_pid_ids = [7, 8, 9]
    syn_labels = [pid_names[i] for i in syn_pid_ids]
    hop_bins = np.arange(0.5, 5.6, 1.0)
    hop_groups = [hop_arr[pid_arr == i] for i in syn_pid_ids]
    axes[1, 1].hist(hop_groups, bins=hop_bins, label=syn_labels, alpha=0.7)
    axes[1, 1].set_title("hop distribution for synergy atoms")
    axes[1, 1].set_xlabel("hop")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_xticks([1, 2, 3, 4])
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.25)

    _savefig(out_dir / "pid_metadata_distributions.png")

    assert np.all(counts == n_per_pid)
    for vals in red_rho_groups:
        uniq = [float(v) for v in np.unique(vals)]
        assert all(any(abs(v - ref) < 1e-3 for ref in (0.2, 0.5, 0.8)) for v in uniq)
    for vals in hop_groups:
        assert set(np.unique(vals).astype(int)).issubset({1, 2, 3, 4})


def test_plot_pid_dependence_distributions_boxplots():
    """
    Distribution of pairwise dependence proxy per pid (repeated mini-batches).
    Produces boxplots that are useful in the MD to show variability, not only means.
    """
    out_dir = _ensure_plot_dir()
    gen = _make_generator(seed=601, sigma=0.45)
    pid_names = all_pid_names()

    repeats = 18
    n_per_repeat = 220
    pairs = [("x1", "x2"), ("x1", "x3"), ("x2", "x3")]
    pair_titles = ["D(1,2)", "D(1,3)", "D(2,3)"]

    scores = np.zeros((3, 10, repeats), dtype=np.float32)
    for p_idx, (a, b) in enumerate(pairs):
        for pid_id in range(10):
            for r in range(repeats):
                batch = _generate_fixed_pid(gen, pid_id=pid_id, n=n_per_repeat)
                scores[p_idx, pid_id, r] = _dependence_proxy(batch[a], batch[b])

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.4), sharex=True)
    colors = ["#3b7ddd", "#d95f02", "#1b9e77"]

    for p_idx, ax in enumerate(axes):
        data = [scores[p_idx, pid_id, :] for pid_id in range(10)]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[p_idx])
            patch.set_alpha(0.45)
        ax.set_title(f"{pair_titles[p_idx]} across pid atoms (distribution over repeated batches)")
        ax.set_ylabel("dependence proxy")
        ax.grid(axis="y", alpha=0.25)

        # Light markers for expected dominant rows.
        if p_idx == 0:
            ax.axvspan(3.6, 4.4, color="gold", alpha=0.12)  # R12 (pid 3, box index 4)
        elif p_idx == 1:
            ax.axvspan(4.6, 5.4, color="gold", alpha=0.12)  # R13
        else:
            ax.axvspan(5.6, 6.4, color="gold", alpha=0.12)  # R23
        ax.axvspan(6.6, 7.4, color="purple", alpha=0.08)  # R123

    axes[-1].set_xticks(np.arange(1, 11))
    axes[-1].set_xticklabels([f"{i}:{name}" for i, name in enumerate(pid_names)], rotation=20, ha="right")
    axes[-1].set_xlabel("pid atom")

    _savefig(out_dir / "pid_dependence_distributions_boxplots.png")

    # Basic signal checks on median behavior.
    med_d12 = np.median(scores[0], axis=1)
    med_d13 = np.median(scores[1], axis=1)
    med_d23 = np.median(scores[2], axis=1)
    assert med_d12[3] > med_d12[0]  # R12 > U1 on pair (1,2)
    assert med_d13[4] > med_d13[0]  # R13 > U1 on pair (1,3)
    assert med_d23[5] > med_d23[0]  # R23 > U1 on pair (2,3)
    assert np.mean([med_d12[6], med_d13[6], med_d23[6]]) > np.mean([med_d12[0], med_d13[0], med_d23[0]])


if __name__ == "__main__":
    # Convenience local runner without pytest.
    test_dataset_shapes_and_metadata()
    test_plot_pairwise_dependence_signatures()
    test_plot_redundancy_monotonicity_vs_rho()
    test_plot_synergy_delta_vs_hop()
    test_plot_atom_gain_controls_ur()
    test_plot_ur_compact_signature_grid_over_sigma()
    test_plot_ur_hyperparameter_sweeps_compact()
    test_plot_ur_intuition_scatter_examples()
    test_plot_r123_pca_all_pairs()
    test_plot_pid_metadata_distributions()
    test_plot_pid_dependence_distributions_boxplots()
    print(f"Saved plots to {PLOT_DIR.resolve()}")
