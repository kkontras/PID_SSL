# PID-SAR-3++: Formal Dataset Specification, Implementation Tutorial, and U/R Intuition Figures

This document has three parts:

1. A formal dataset/task specification written in a paper-style format.
2. A code tutorial mapping the specification to the implementation.
3. Explanatory U/R figures (including PCA-based scatter plots) with interpretation.

Reference implementation:

- `pid_sar3_dataset.py`
- `tests/test_pid_sar3_dataset.py`

## 1. Formal Dataset and Task Specification

### 1.1 Dataset Overview and Notation

PID-SAR-3++ is a synthetic three-view benchmark for evaluating multi-view representation learning under controlled information structure. Each sample consists of three observations $x_1, x_2, x_3 \in \mathbb{R}^d$.

Exactly one PID-inspired information atom is active per sample. The atom families are Unique (`U`), Redundancy (`R`), and Directional Synergy (`S`), and the full atom set is $\mathcal{A} = \{U_1,U_2,U_3,R_{12},R_{13},R_{23},R_{123},S_{12 \to 3},S_{13 \to 2},S_{23 \to 1}\}$.

Each sample is annotated with a categorical atom identifier `pid_id ∈ {0,…,9}` used for evaluation only (not for SSL training).

The generator returns $(x_1,x_2,x_3,\mathrm{pid\_id},\alpha,\sigma,\rho,h)$,

where `alpha` is the signal amplitude, `sigma` is the observation noise scale, `rho` is the redundancy overlap parameter (undefined for non-redundancy atoms; encoded as `-1`), and `h` is the synergy depth parameter (undefined for non-synergy atoms; encoded as `0`).

### 1.2 Task Definition (Training and Evaluation Protocol)

Training protocol:

- the learner receives only the three views `(x1, x2, x3)`,
- the learner does not receive `pid_id`, `rho`, or `h`,
- a multi-view self-supervised objective is trained on the observations.

Evaluation protocol:

- encoders are frozen after training,
- representations are evaluated for retention of unique, redundant, and directional-synergistic structure.

Generator-level validation (before training encoders):

- evaluate raw observations using dependence and synergy proxies,
- verify that signatures match the intended atom-level structure.

This note emphasizes the `U/R` subset first because it provides the most interpretable sanity checks.

### 1.3 Generative Parameters

The latent dimensionality satisfies `m << d`. Typical defaults are $d = 32$, $m = 8$, $\alpha \sim \mathrm{Uniform}(\alpha_{\min}, \alpha_{\max})$, $\sigma > 0$, and $(\rho,h) \in \mathcal{R}\times\mathcal{H}$ with $\mathcal{R}\subset (0,1)$ and $\mathcal{H}\subset \mathbb{N}$.

Default values in `pid_sar3_dataset.py`:

- `alpha_min = 0.8`, `alpha_max = 1.2`
- `R = {0.2, 0.5, 0.8}` (implemented as `rho_choices`)
- `H = {1,2,3,4}` (implemented as `hop_choices`)

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

Expected `U/R` signatures:

- `U1/U2/U3`: low pairwise dependence (near noise floor),
- `R12/R13/R23`: high dependence only for the matching pair,
- `R123`: elevated dependence across all three pairs,
- dependence increases with `rho`,
- dependence decreases as `sigma` increases.

### 2.2 PCA-Based Geometric Intuition

For a fixed atom, let `X_k` denote the matrix of samples from view `k`, and define the first principal-component score as $z_k = \mathrm{PC1}(X_k)$ for $k\in\{1,2,3\}$.

Scatter plots of `(z_i, z_j)` provide a qualitative geometric diagnostic:

- unique atoms produce diffuse, weakly structured clouds,
- redundant atoms produce aligned / elongated clouds due to shared latent structure.

This diagnostic is especially useful for presentation and sanity checking because it makes the latent dependence geometry visually explicit.

## 3. Code Tutorial (How the Dataset Is Implemented and Used)

This section maps the formal definition to the actual code.

### 3.1 Instantiate the Generator

`PIDSar3DatasetGenerator` encapsulates:

- fixed projection sampling,
- fixed synergy MLP sampling,
- de-leakage fitting,
- per-sample generation and batch generation.

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

### 3.2 Generate a Single Sample

```python
sample = gen.sample(pid_id=3)  # R12

# keys: x1, x2, x3, pid_id, alpha, sigma, rho, hop
print(sample["x1"].shape)  # (32,)
print(sample["pid_id"])    # 3
```

### 3.3 Generate a Balanced U/R Subset

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

### 3.4 Save the Dataset to Disk

```python
import numpy as np
np.savez_compressed("data/pid_sar3_ur_train.npz", **batch)
```

### 3.5 Where the Diagnostics Are Implemented

The main U/R plots are produced by these test functions in `tests/test_pid_sar3_dataset.py`:

- `test_plot_ur_compact_signature_grid_over_sigma()`
- `test_plot_ur_hyperparameter_sweeps_compact()`
- `test_plot_ur_intuition_scatter_examples()`

These functions are written as tests so they can serve both as regression checks and as reproducible figure generation scripts.

## 4. Commands to Reproduce the Dataset and Figures

### 4.1 Generate the U/R Diagnostic Figures (Recommended Entry Point)

```bash
python - <<'PY'
from tests.test_pid_sar3_dataset import (
    test_plot_pid_metadata_distributions,
    test_plot_pid_dependence_distributions_boxplots,
    test_plot_ur_compact_signature_grid_over_sigma,
    test_plot_ur_hyperparameter_sweeps_compact,
    test_plot_ur_intuition_scatter_examples,
)

test_plot_pid_metadata_distributions()
test_plot_pid_dependence_distributions_boxplots()
test_plot_ur_compact_signature_grid_over_sigma()
test_plot_ur_hyperparameter_sweeps_compact()
test_plot_ur_intuition_scatter_examples()
print("Saved plots under test_outputs/pid_sar3")
PY
```

Output figures:

- `test_outputs/pid_sar3/pid_metadata_distributions.png`
- `test_outputs/pid_sar3/pid_dependence_distributions_boxplots.png`
- `test_outputs/pid_sar3/ur_compact_signature_grid_over_sigma.png`
- `test_outputs/pid_sar3/ur_hyperparameter_sweeps_compact.png`
- `test_outputs/pid_sar3/ur_intuition_scatter_examples.png`

### 4.2 Generate and Save a Balanced U/R Dataset (`.npz`)

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

### 4.3 Generate Train / Val / Test Splits

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

## 5. Explanatory Figures and Intuitive Analysis (U/R Focus)

This section explains the key concepts using the generated figures.

### 5.1 Figure A: PID Metadata Distributions (Sanity Check)

File:

- `test_outputs/pid_sar3/pid_metadata_distributions.png`

![PID metadata distributions](test_outputs/pid_sar3/pid_metadata_distributions.png)

What it shows:

- balanced counts across all `pid_id` values (when using a balanced generation schedule),
- per-`pid` distributions of `alpha`,
- `rho` support only for redundancy atoms (`R12`, `R13`, `R23`, `R123`),
- `hop` support only for synergy atoms (`S12->3`, `S13->2`, `S23->1`).

Why this matters:

- it verifies that the generator metadata are sampled as intended and that atom-specific parameters are activated only in the relevant atom families.

### 5.2 Figure B: PID Dependence Distributions (Repeated-Batch Variability)

File:

- `test_outputs/pid_sar3/pid_dependence_distributions_boxplots.png`

![PID dependence distributions](test_outputs/pid_sar3/pid_dependence_distributions_boxplots.png)

What it shows:

- boxplots of pairwise dependence proxies across repeated mini-batches for each `pid`,
- variability around the expected signatures, not just mean values.

How to read it:

- in `D(1,2)`, `R12` should dominate and `R123` should also be elevated,
- in `D(1,3)`, `R13` should dominate and `R123` should also be elevated,
- in `D(2,3)`, `R23` should dominate and `R123` should also be elevated,
- `U1/U2/U3` should stay near the noise floor in all three panels.

Why this matters:

- it demonstrates robustness of the intended dependence topology across repeated sampling, which is useful for substantiating claims in the text.

### 5.3 Figure C: U/R Signature Grid Across Noise

File:

- `test_outputs/pid_sar3/ur_compact_signature_grid_over_sigma.png`

![U/R signature grid over sigma](test_outputs/pid_sar3/ur_compact_signature_grid_over_sigma.png)

What it shows:

- Rows are atoms (`U1,U2,U3,R12,R13,R23,R123`).
- Columns are pairwise dependence proxies (`D(1,2)`, `D(1,3)`, `D(2,3)`).
- Three panels correspond to different noise levels `sigma`.

How to read it:

- `U1/U2/U3` rows should remain low across all pairs (no shared signal).
- `R12` should be bright mainly in `D(1,2)`.
- `R13` should be bright mainly in `D(1,3)`.
- `R23` should be bright mainly in `D(2,3)`.
- `R123` should be elevated in all three columns.
- As `sigma` increases, all values typically compress toward the noise floor.

Why this is useful:

- It is a single compact sanity-check that immediately reveals whether the generator is producing the intended dependence topology.

### 5.4 Figure D: Hyperparameter Sweeps (`rho`, `sigma`, `alpha`)

File:

- `test_outputs/pid_sar3/ur_hyperparameter_sweeps_compact.png`

![U/R hyperparameter sweeps](test_outputs/pid_sar3/ur_hyperparameter_sweeps_compact.png)

Left panel (redundancy overlap):

- For `R12`, the dependence proxy $D(x_1,x_2)$ should increase with $\rho$.
- Multiple curves at different `sigma` show that noise weakens observed dependence but preserves the monotonic trend.

Right panel (signal amplitude and noise):

- Mean $\|x_1\|_2$ increases as `alpha` increases.
- Larger `sigma` shifts norms upward and broadens the effective scale (signal + noise).
- Comparing `U1` and `R123` shows the effect under two different atom structures.

Why this is useful:

- It ties abstract hyperparameters to direct geometric/statistical effects in the observed data.

### 5.5 Figure E: PCA Intuition Scatter Plots (What You Liked)

File:

- `test_outputs/pid_sar3/ur_intuition_scatter_examples.png`

![U/R PCA intuition scatter plots](test_outputs/pid_sar3/ur_intuition_scatter_examples.png)

Construction:

- For each atom (`U1`, `R12`, `R123`) and noise level (`sigma=0.15`, `sigma=0.9`), compute:
  - `PC1(x1)` across samples,
  - `PC1(x2)` across samples,
  - scatter the paired scores for the same datapoints.

Interpretation:

- `U1`:
  - signal is only in view 1,
  - `x2` is mostly noise,
  - the scatter is diffuse with weak structure.
- `R12`:
  - both views contain overlapping latent structure,
  - the scatter becomes elongated/aligned,
  - stronger apparent coupling is visible.
- `R123`:
  - `x1` and `x2` both inherit a shared latent factor (plus view-specific perturbation),
  - alignment remains visible, though the geometry differs from `R12`.
- Increasing `sigma`:
  - broadens all point clouds,
  - weakens visible alignment and correlation magnitude.

Important caveat (PCA sign ambiguity):

- The sign of a principal component is arbitrary, so the slope can flip between runs.
- The presence/absence of alignment and the magnitude of association are the meaningful signals.

## 6. Suggested Paper-Style Writeup Snippets

### 6.1 Methods (Dataset)

We generate a synthetic three-view dataset in which each sample contains exactly one information atom from a predefined PID-inspired atom set. Let $x_1,x_2,x_3 \in \mathbb{R}^d$ denote the observed views. For each view and atom-specific component, we sample a fixed projection matrix $P_k^{(c)} \in \mathbb{R}^{d\times m}$ at dataset initialization. Unique atoms are generated by projecting a latent Gaussian vector into a single view, while redundant atoms are generated by mixing a shared latent Gaussian factor with view-specific Gaussian perturbations using an overlap coefficient $\rho$. All observations include additive isotropic Gaussian noise with scale $\sigma$, and signal amplitude is modulated by a per-sample scalar $\alpha$. This construction provides explicit control over dependence topology (unique, pairwise redundancy, and triple redundancy) and signal-to-noise conditions.

### 6.2 Results (U/R Sanity Checks)

On the U/R subset, pairwise dependence heatmaps recover the intended atom structure: unique atoms remain near the noise floor across all pairwise dependencies, pairwise redundant atoms activate the corresponding view pair, and triple redundancy elevates all pairwise dependencies. PCA-based scatter plots of $(\mathrm{PC1}(x_1),\mathrm{PC1}(x_2))$ provide an intuitive geometric corroboration: unique atoms appear as diffuse clouds, whereas redundant atoms exhibit aligned low-dimensional structure whose visibility degrades gracefully as the observation noise $\sigma$ increases.
