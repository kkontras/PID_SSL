# PID-SAR-3++: Formal Dataset Specification, Implementation Tutorial, and U/R Intuition Figures

This document has four parts: a formal dataset/task specification written in a paper-style format, validation metrics and intuition for the raw observations, a dataset-exploration section with figures and equation-linked interpretation, and a code tutorial mapping the specification to the implementation. The reference implementation is in `pid_sar3_dataset.py` and `tests/test_pid_sar3_dataset.py`.

## 1. Formal Dataset and Task Specification

### 1.1 Dataset Overview and Notation

PID-SAR-3++ is a synthetic three-view benchmark for evaluating multi-view representation learning under controlled information structure. Each sample consists of three observations $x_1, x_2, x_3 \in \mathbb{R}^d$.

Exactly one PID-inspired information atom is active per sample. The atom families are Unique (`U`), Redundancy (`R`), and Directional Synergy (`S`), and the full atom set is $\mathcal{A} = \{U_1,U_2,U_3,R_{12},R_{13},R_{23},R_{123},S_{12 \to 3},S_{13 \to 2},S_{23 \to 1}\}$.

Each sample is annotated with a categorical atom identifier `pid_id ∈ {0,…,9}` used for evaluation only (not for SSL training).

The generator returns $(x_1,x_2,x_3,\mathrm{pid\_id},\alpha,\sigma,\rho,h)$,

where `alpha` is the signal amplitude, `sigma` is the observation noise scale, `rho` is the redundancy overlap parameter (undefined for non-redundancy atoms; encoded as `-1`), and `h` is the synergy depth parameter (undefined for non-synergy atoms; encoded as `0`).

### 1.2 Task Definition (Training and Evaluation Protocol)

In the training protocol, the learner receives only the three views `(x1, x2, x3)` and does not receive `pid_id`, `rho`, or `h`; a multi-view self-supervised objective is trained directly on the observations. In the evaluation protocol, encoders are frozen after training and the resulting representations are assessed for retention of unique, redundant, and directional-synergistic structure. Before any encoder is trained, the generator itself should be validated on the raw observations using dependence and synergy proxies to verify that the empirical signatures match the intended atom-level structure. This note emphasizes the `U/R` subset first because it provides the most interpretable sanity checks.

### 1.3 Generative Parameters

The latent dimensionality satisfies `m << d`. Typical defaults are $d = 32$, $m = 8$, $\alpha \sim \mathrm{Uniform}(\alpha_{\min}, \alpha_{\max})$, $\sigma > 0$, and $(\rho,h) \in \mathcal{R}\times\mathcal{H}$ with $\mathcal{R}\subset (0,1)$ and $\mathcal{H}\subset \mathbb{N}$.

Default values in `pid_sar3_dataset.py` are `alpha_min = 0.8`, `alpha_max = 1.2`, `rho_choices = {0.2, 0.5, 0.8}`, and `hop_choices = {1,2,3,4}`.

### 1.4 Fixed Projection Operators (Sampled Once per Dataset Seed)

For each view `k ∈ {1,2,3}` and each component `c`, the generator samples a fixed projection matrix $P_k^{(c)} \in \mathbb{R}^{d \times m}$ with entries $P_k^{(c)}[i,j] \sim \mathcal{N}(0,1/d)$.

Each column is normalized as $P_k^{(c)}[:,j] \leftarrow P_k^{(c)}[:,j]/\|P_k^{(c)}[:,j]\|_2$.

These operators are held fixed for all samples generated with the same dataset seed.

### 1.5 Observation Noise

Each view receives additive isotropic Gaussian noise $\varepsilon_k \sim \mathcal{N}(0,\sigma^2 I_d)$ for $k\in\{1,2,3\}$, and the observed variable is $x_k = \mathrm{signal}_k + \varepsilon_k$.

### 1.6 Unique Atoms

For `U_i`, sample a latent Gaussian vector $u \sim \mathcal{N}(0,I_m)$.

The active view receives the projected latent, while inactive views contain noise only: $x_i = \alpha P_i^{(U_i)} u + \varepsilon_i$ and $x_j = \varepsilon_j$ for $j \neq i$.

### 1.7 Pairwise Redundancy Atoms

For `R_{ij}`, sample $r,\eta_i,\eta_j \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,I_m)$.

Construct view-specific latent realizations with overlap coefficient `rho`: $r_i = \sqrt{\rho}\,r + \sqrt{1-\rho}\,\eta_i$ and $r_j = \sqrt{\rho}\,r + \sqrt{1-\rho}\,\eta_j$.

Then generate the observations $x_i = \alpha P_i^{(R_{ij})} r_i + \varepsilon_i$ and $x_j = \alpha P_j^{(R_{ij})} r_j + \varepsilon_j$, while $x_k = \varepsilon_k$ for $k\notin\{i,j\}$.

As `rho` increases, shared structure between the active views becomes stronger.

### 1.8 Triple Redundancy Atom

For `R_{123}`, sample $r,\eta_1,\eta_2,\eta_3 \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0,I_m)$.

Define per-view redundant latents $r_k = \sqrt{\rho}\,r + \sqrt{1-\rho}\,\eta_k$ for $k\in\{1,2,3\}$, and set $x_k = \alpha P_k^{(R_{123})} r_k + \varepsilon_k$.

### 1.9 Directional Synergy Atoms

For `S_{ij→k}`, sample source latents and a hop parameter, $a,b \sim \mathcal{N}(0,I_m)$ and $h \in \mathcal{H}$.

Source views are generated linearly as $x_i = \alpha P_i^{(A_{ij})} a + \varepsilon_i$ and $x_j = \alpha P_j^{(B_{ij})} b + \varepsilon_j$.

A fixed nonlinear readout network `phi_h` produces a target latent $s_0 = \phi_h([a,b]) \in \mathbb{R}^m$. The latent is de-leaked via $s = s_0 - C_a^{(h)} a - C_b^{(h)} b$, and the target view is generated as $x_k = \alpha P_k^{(\mathrm{SYN}_{ij})} s + \varepsilon_k$.

This construction reduces single-source linear leakage and yields a more directional synergy signal.

### 1.10 Synergy De-leakage Fit (Offline, per Dataset Seed)

For each hop `h`, de-leakage maps are fit by ridge regression on synthetic latent samples:

```math
W^{(h)} = \arg\min_W \|S_0 - XW\|_F^2 + \lambda \|W\|_F^2,
```

with

```math
X = [A\;B] \in \mathbb{R}^{N\times 2m},\qquad S_0 \in \mathbb{R}^{N\times m}.
```

The fitted matrix is partitioned as:

```math
W^{(h)} =
\begin{bmatrix}
C_a^{(h)} \\
C_b^{(h)}
\end{bmatrix}.
```

These maps are then used during generation to compute the de-leaked target latent `s`.

## 2. Validation Metrics (Raw Data, Pre-Encoder)

### 2.1 Symmetric Dependence Proxy

Given two view matrices `X_A` and `X_B` (rows are samples), define $D(X_A, X_B)=\frac{1}{2}\left(R^2(X_A\to X_B) + R^2(X_B\to X_A)\right)$,

where each `R^2` is computed using ridge regression on a train/test split.

Intuitively, `D(1,2)` answers the question: "How much predictable structure is shared between view 1 and view 2?" If `x1` helps linearly predict `x2` (and conversely `x2` helps linearly predict `x1`), then `D(1,2)` is high; if the two views mostly contain unrelated signal/noise, then `D(1,2)` is low. In this dataset, `U1` should have low `D(1,2)` because only `x1` contains signal and `x2` is mostly noise, `R12` should have high `D(1,2)` because both views share overlapping latent structure, and `R123` should also have elevated `D(1,2)` because views 1 and 2 both inherit the shared triple-redundant latent. In paper terms, `D(1,2)` is not a causal quantity and not a PID estimator; it is a controlled dependence proxy used to validate whether the generator induces the intended cross-view geometry in the observations. Accordingly, the expected U/R signatures are low pairwise dependence for `U1/U2/U3`, pair-specific high dependence for `R12/R13/R23`, broad elevation for `R123`, monotonic growth with `rho`, and degradation as `sigma` increases.

### 2.2 PCA-Based Geometric Intuition

For a fixed atom, let `X_k` denote the matrix of samples from view `k`, and define the first principal-component score as $z_k = \mathrm{PC1}(X_k)$ for $k\in\{1,2,3\}$.

Scatter plots of `(z_i, z_j)` provide a qualitative geometric diagnostic: unique atoms produce diffuse, weakly structured clouds, while redundant atoms produce aligned or elongated clouds due to shared latent structure. This diagnostic is especially useful for presentation and sanity checking because it makes the latent dependence geometry visually explicit in a way that complements scalar dependence summaries such as `D(1,2)`.

## 3. Dataset Exploration (Figures with Equations and Interpretation)

This section is intentionally placed before the code tutorial so that the reader first understands what the dataset looks like statistically and geometrically. The central quantity used throughout is the symmetric dependence proxy $D(X_A, X_B)=\tfrac{1}{2}(R^2(X_A\to X_B)+R^2(X_B\to X_A))$, which summarizes how much linearly predictable structure is shared across two views on held-out samples. In the U/R regime, `D(1,2)` should be low for unique atoms and high for atoms that explicitly share signal between views 1 and 2.

### 3.1 Figure A: Atom Gain Controls (Amplifying U vs R Unequally)

![Atom gain controls for U/R](test_outputs/pid_sar3/atom_gain_controls_ur.png)

This figure demonstrates a controllable extension of the generator in which the effective signal amplitude is modulated by an atom-dependent gain, i.e., $\alpha_{\mathrm{eff}}=\alpha \cdot g(\mathrm{pid\_id})$, while the observation noise scale $\sigma$ is held fixed. The left panel shows how this changes the observed scale (mean $\|x_1\|$), and the right panel shows how it changes the cross-view dependence proxy `D(1,2)`. The key point is that boosting redundancy (e.g., `redundancy_gain > 1`) selectively increases `D(1,2)` for `R12`, whereas boosting unique atoms primarily increases magnitude without inducing cross-view dependence. Per-`pid` overrides allow unequal emphasis within the same family (for example, stronger `R12` but weaker `R123`), which is useful for creating difficulty-controlled stress tests.

### 3.2 Figure B: PID Metadata Distributions (Sampling Sanity Check)

![PID metadata distributions](test_outputs/pid_sar3/pid_metadata_distributions.png)

This figure validates the sampling pipeline itself. Under a balanced generation schedule, class counts should be uniform across `pid_id`. The per-`pid` `alpha` boxplots should follow the configured amplitude range, while `rho` should appear only for redundancy atoms and `hop` should appear only for synergy atoms. This is important because many downstream conclusions rely on the assumption that metadata are activated only when the corresponding generative mechanism is active.

### 3.3 Figure C: PID Dependence Distributions (Repeated-Batch Variability)

![PID dependence distributions](test_outputs/pid_sar3/pid_dependence_distributions_boxplots.png)

Rather than showing only a single estimate per atom, this figure shows repeated-batch distributions of the dependence proxy for each pair of views. The governing quantity is again $D(i,j)$, and the figure makes two things visible simultaneously: the expected ordering (e.g., `R12` should dominate in `D(1,2)`, `R13` in `D(1,3)`, `R23` in `D(2,3)`) and the sampling variability around those expectations. `R123` should remain elevated across all three panels because the same triple-redundant latent contributes to all views.

### 3.4 Figure D: U/R Signature Grid Across Noise

![U/R signature grid over sigma](test_outputs/pid_sar3/ur_compact_signature_grid_over_sigma.png)

This compact heatmap view is the quickest way to inspect the U/R subset. Each cell corresponds to a dependence score $D(i,j)$ for a fixed atom and a fixed noise level $\sigma$. Unique atoms (`U1`, `U2`, `U3`) should remain near the noise floor because only one view carries signal, whereas pairwise redundancy atoms should activate the matching pair and `R123` should elevate all pairs. As $\sigma$ increases, all dependence values typically contract toward zero, which is exactly the expected degradation under additive noise.

### 3.5 Figure E: Hyperparameter Sweeps (`rho`, `sigma`, `alpha`)

![U/R hyperparameter sweeps](test_outputs/pid_sar3/ur_hyperparameter_sweeps_compact.png)

The left panel links the redundancy mechanism directly to the data statistics: because $r_i=\sqrt{\rho}\,r+\sqrt{1-\rho}\,\eta_i$ and $r_j=\sqrt{\rho}\,r+\sqrt{1-\rho}\,\eta_j$, increasing $\rho$ increases shared latent content and should therefore increase $D(x_1,x_2)$ for `R12`. The right panel shows how the observed norm scales with signal amplitude and noise. Since observations take the form $x_k=\mathrm{signal}_k+\varepsilon_k$, increasing `alpha` increases the signal contribution, while increasing `sigma` increases the noise contribution and changes the effective scale of the raw vectors.

### 3.6 Figure F: PCA Intuition Scatter Plots

![U/R PCA intuition scatter plots](test_outputs/pid_sar3/ur_intuition_scatter_examples.png)

These plots visualize the first principal-component scores $z_k=\mathrm{PC1}(X_k)$ and scatter paired scores $(z_1,z_2)$ for the same samples. They are not estimators of PID quantities, but they are highly interpretable geometric diagnostics. The useful reading strategy is to compare the *shape* of the cloud and the *magnitude* of the panel correlation, not the sign of the slope. In the currently generated figure (same code path and seeds as the test), the low-noise row (`sigma = 0.15`) shows a near-zero association for `U1` (approximately $|r| \approx 0.016$), while `R12` and `R123` show visibly stronger alignment (approximately $|r| \approx 0.218$ and $|r| \approx 0.274$, respectively). In the high-noise row (`sigma = 0.9`), the `R12` panel still retains a visible dependence signal (approximately $|r| \approx 0.137$), whereas `U1` remains near zero (approximately $|r| \approx 0.003$) and `R123` may collapse toward the noise floor in this particular projection/view pair (approximately $|r| \approx 0.007$), which is a useful reminder that PCA is a low-dimensional diagnostic of one pair of views rather than a complete summary of the atom. Because PCA signs are arbitrary, slope direction may flip across runs; the presence and magnitude of alignment are the meaningful features.

## 4. Code Tutorial (How the Dataset Is Implemented and Used)

This section maps the formal definition to the actual code.

### 4.1 Instantiate the Generator

`PIDSar3DatasetGenerator` encapsulates fixed projection sampling, fixed synergy MLP sampling, de-leakage fitting, and per-sample / batch generation.

Minimal example:

```python
from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator

cfg = PIDDatasetConfig(
    d=32,
    m=8,
    sigma=0.45,
    alpha_min=0.8,
    alpha_max=1.2,
    rho_choices=(0.2, 0.5, 0.8),
    hop_choices=(1, 2, 3, 4),
    seed=0,
)
gen = PIDSar3DatasetGenerator(cfg)
```

To amplify atom families (or specific atoms) unequally, use gain controls:

```python
cfg = PIDDatasetConfig(
    seed=0,
    unique_gain=1.6,       # boost all U atoms
    redundancy_gain=0.8,   # suppress all R atoms
    synergy_gain=1.0,
    pid_gain_overrides={
        3: 2.0,  # specifically boost R12
        6: 0.7,  # specifically weaken R123
    },
)
gen = PIDSar3DatasetGenerator(cfg)
```

The effective signal amplitude becomes `alpha_eff = alpha * gain(pid_id)`, while the additive noise scale `sigma` is unchanged.

### 4.2 Generate a Single Sample

```python
sample = gen.sample(pid_id=3)  # R12

# keys: x1, x2, x3, pid_id, alpha, sigma, rho, hop
print(sample["x1"].shape)  # (32,)
print(sample["pid_id"])    # 3
```

### 4.3 Generate a Balanced U/R Subset

The U/R-only subset corresponds to $\{0,1,2,3,4,5,6\} = \{U_1,U_2,U_3,R_{12},R_{13},R_{23},R_{123}\}$.

```python
import numpy as np

ur_pid_ids = [0, 1, 2, 3, 4, 5, 6]
n_per_atom = 5000
pid_schedule = np.repeat(ur_pid_ids, n_per_atom)

batch = gen.generate(n=len(pid_schedule), pid_ids=pid_schedule.tolist())
print(batch["x1"].shape)      # (35000, d)
print(batch["pid_id"].shape)  # (35000,)
```

### 4.4 Save the Dataset to Disk

```python
import numpy as np
np.savez_compressed("data/pid_sar3_ur_train.npz", **batch)
```

### 4.5 Where the Diagnostics Are Implemented

The main U/R plots are produced by `test_plot_atom_gain_controls_ur()`, `test_plot_pid_metadata_distributions()`, `test_plot_pid_dependence_distributions_boxplots()`, `test_plot_ur_compact_signature_grid_over_sigma()`, `test_plot_ur_hyperparameter_sweeps_compact()`, and `test_plot_ur_intuition_scatter_examples()` in `tests/test_pid_sar3_dataset.py`. These functions are written as tests so they can serve both as regression checks and as reproducible figure-generation scripts.

## 5. Commands to Reproduce the Dataset and Figures

### 5.1 Generate the U/R Diagnostic Figures (Recommended Entry Point)

```bash
python - <<'PY'
from tests.test_pid_sar3_dataset import (
    test_plot_atom_gain_controls_ur,
    test_plot_pid_metadata_distributions,
    test_plot_pid_dependence_distributions_boxplots,
    test_plot_ur_compact_signature_grid_over_sigma,
    test_plot_ur_hyperparameter_sweeps_compact,
    test_plot_ur_intuition_scatter_examples,
)

test_plot_atom_gain_controls_ur()
test_plot_pid_metadata_distributions()
test_plot_pid_dependence_distributions_boxplots()
test_plot_ur_compact_signature_grid_over_sigma()
test_plot_ur_hyperparameter_sweeps_compact()
test_plot_ur_intuition_scatter_examples()
print("Saved plots under test_outputs/pid_sar3")
PY
```

This command generates `test_outputs/pid_sar3/atom_gain_controls_ur.png`, `test_outputs/pid_sar3/pid_metadata_distributions.png`, `test_outputs/pid_sar3/pid_dependence_distributions_boxplots.png`, `test_outputs/pid_sar3/ur_compact_signature_grid_over_sigma.png`, `test_outputs/pid_sar3/ur_hyperparameter_sweeps_compact.png`, and `test_outputs/pid_sar3/ur_intuition_scatter_examples.png`.

### 5.2 Generate and Save a Balanced U/R Dataset (`.npz`)

```bash
mkdir -p data
python - <<'PY'
import numpy as np
from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator

cfg = PIDDatasetConfig(seed=0, d=32, m=8, sigma=0.45)
gen = PIDSar3DatasetGenerator(cfg)

ur_pid_ids = [0, 1, 2, 3, 4, 5, 6]
n_per_atom = 5000
pid_schedule = np.repeat(ur_pid_ids, n_per_atom)

batch = gen.generate(n=len(pid_schedule), pid_ids=pid_schedule.tolist())
np.savez_compressed("data/pid_sar3_ur_train.npz", **batch)
print("Saved data/pid_sar3_ur_train.npz with", len(pid_schedule), "samples")
PY
```

### 5.3 Generate a U/R Dataset with Intentional U/R Imbalance (Gain Controls)

```bash
mkdir -p data
python - <<'PY'
import numpy as np
from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator

cfg = PIDDatasetConfig(
    seed=7,
    d=32,
    m=8,
    sigma=0.45,
    unique_gain=1.5,
    redundancy_gain=0.9,
    pid_gain_overrides={3: 2.0, 6: 0.6},  # stronger R12, weaker R123
)
gen = PIDSar3DatasetGenerator(cfg)

ur_pid_ids = [0, 1, 2, 3, 4, 5, 6]
pid_schedule = np.repeat(ur_pid_ids, 3000)
batch = gen.generate(n=len(pid_schedule), pid_ids=pid_schedule.tolist())
np.savez_compressed("data/pid_sar3_ur_imbalanced_gain.npz", **batch)
print("Saved data/pid_sar3_ur_imbalanced_gain.npz")
PY
```

### 5.4 Generate Train / Val / Test Splits

```bash
mkdir -p data
python - <<'PY'
import numpy as np
from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator

cfg = PIDDatasetConfig(seed=42, d=32, m=8, sigma=0.45)
gen = PIDSar3DatasetGenerator(cfg)
ur_pid_ids = [0,1,2,3,4,5,6]

def make_split(path, n_per_atom):
    pid_schedule = np.repeat(ur_pid_ids, n_per_atom)
    batch = gen.generate(n=len(pid_schedule), pid_ids=pid_schedule.tolist())
    np.savez_compressed(path, **batch)
    print("Saved", path, "N=", len(pid_schedule))

make_split("data/pid_sar3_ur_train.npz", 10000)
make_split("data/pid_sar3_ur_val.npz",   1000)
make_split("data/pid_sar3_ur_test.npz",  1000)
PY
```

## 6. Suggested Paper-Style Writeup Snippets

### 6.1 Methods (Dataset)

We generate a synthetic three-view dataset in which each sample contains exactly one information atom from a predefined PID-inspired atom set. Let $x_1,x_2,x_3 \in \mathbb{R}^d$ denote the observed views. For each view and atom-specific component, we sample a fixed projection matrix $P_k^{(c)} \in \mathbb{R}^{d\times m}$ at dataset initialization. Unique atoms are generated by projecting a latent Gaussian vector into a single view, while redundant atoms are generated by mixing a shared latent Gaussian factor with view-specific Gaussian perturbations using an overlap coefficient $\rho$. All observations include additive isotropic Gaussian noise with scale $\sigma$, and signal amplitude is modulated by a per-sample scalar $\alpha$. This construction provides explicit control over dependence topology (unique, pairwise redundancy, and triple redundancy) and signal-to-noise conditions.

### 6.2 Results (U/R Sanity Checks)

On the U/R subset, pairwise dependence heatmaps recover the intended atom structure: unique atoms remain near the noise floor across all pairwise dependencies, pairwise redundant atoms activate the corresponding view pair, and triple redundancy elevates all pairwise dependencies. PCA-based scatter plots of $(\mathrm{PC1}(x_1),\mathrm{PC1}(x_2))$ provide an intuitive geometric corroboration: unique atoms appear as diffuse clouds, whereas redundant atoms exhibit aligned low-dimensional structure whose visibility degrades gracefully as the observation noise $\sigma$ increases.
