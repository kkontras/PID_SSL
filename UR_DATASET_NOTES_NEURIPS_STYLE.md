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

PID-SAR-3++ is a synthetic three-view benchmark for evaluating multi-view representation learning under controlled information structure. Each sample contains three observations

$$
x_1, x_2, x_3 \in \mathbb{R}^d,
$$

and is generated so that **exactly one PID-inspired information atom** is active per sample. The atom families are:

$$
\text{Unique} \ (U), \qquad \text{Redundancy} \ (R), \qquad \text{Directional Synergy} \ (S).
$$

The full atom set is:

$$
\mathcal{A} = \{U_1,U_2,U_3,R_{12},R_{13},R_{23},R_{123},S_{12 \to 3},S_{13 \to 2},S_{23 \to 1}\}.
$$

Each sample is annotated with a categorical atom identifier $\mathrm{pid\_id}\in\{0,\dots,9\}$ (used for evaluation only, not for SSL training).

The generator returns the tuple:

$$
(x_1,x_2,x_3,\texttt{pid\_id},\alpha,\sigma,\rho,h),
$$

where $\alpha$ is the signal amplitude, $\sigma$ is the observation noise scale, $\rho$ is the redundancy overlap parameter (undefined for non-redundancy atoms; represented as `-1` in code), and $h$ is the synergy depth parameter (undefined for non-synergy atoms; represented as `0` in code).

### 1.2 Task Definition and Evaluation Objective

The benchmark is designed for the following training/evaluation protocol.

Training input:

- the learner observes only $(x_1,x_2,x_3)$;
- the learner does not receive $\mathrm{pid\_id}$, $\rho$, or $h$.

Evaluation objective:

- train a multi-view self-supervised objective on the generated observations;
- freeze the learned encoders;
- evaluate whether the resulting representations preserve the intended information structure (unique, redundant, and directional synergistic components).

Generator validation (pre-model):

- prior to encoder training, one validates the generator directly on raw observations using dependence and synergy proxy metrics;
- in this document, the emphasis is on the U/R subset because it provides the simplest and most interpretable sanity checks.

### 1.3 Generative Parameters

The latent dimensionality satisfies $m \ll d$. Typical defaults used in the current implementation are:

$$
d = 32,\qquad m = 8,
$$
$$
\alpha \sim \mathrm{Uniform}(\alpha_{\min}, \alpha_{\max}),\qquad \sigma > 0,
$$
$$
\rho \in \mathcal{R}\subset(0,1),\qquad h \in \mathcal{H}\subset\mathbb{N}.
$$

Default values in `pid_sar3_dataset.py`:

- $\alpha_{\min}=0.8,\ \alpha_{\max}=1.2$
- $\mathcal{R}=\{0.2, 0.5, 0.8\}$
- $\mathcal{H}=\{1,2,3,4\}$

### 1.4 Fixed Projection Operators (Sampled Once Per Dataset Seed)

For each view $k \in \{1,2,3\}$ and each component $c$, a fixed projection matrix is sampled:

$$
P_k^{(c)} \in \mathbb{R}^{d \times m},
\qquad
P_k^{(c)}[i,j] \sim \mathcal{N}(0,1/d).
$$

Columns are normalized:

$$
P_k^{(c)}[:,j] \leftarrow \frac{P_k^{(c)}[:,j]}{\left\|P_k^{(c)}[:,j]\right\|_2}.
$$

These matrices are sampled once per dataset seed and then held fixed. Consequently, inter-sample variation is attributable only to latent draws, amplitudes, and observation noise.

### 1.5 Observation Noise

All views receive additive isotropic Gaussian noise:

$$
\varepsilon_k \sim \mathcal{N}(0,\sigma^2 I_d), \qquad k\in\{1,2,3\}.
$$

The final observation is always of the form:

$$
x_k = \text{signal}_k + \varepsilon_k.
$$

### 1.6 Unique Atoms

For $U_i$, sample

$$
u \sim \mathcal{N}(0,I_m),
$$

and define

$$
x_i = \alpha P_i^{(U_i)} u + \varepsilon_i,
$$
$$
x_j = \varepsilon_j \quad \text{for } j\neq i.
$$

This creates signal in exactly one view and noise-only observations in the other two.

### 1.7 Pairwise Redundancy Atoms

For $R_{ij}$, sample

$$
r,\eta_i,\eta_j \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,I_m),
$$

and define noisy shared latents

$$
r_i = \sqrt{\rho}\,r + \sqrt{1-\rho}\,\eta_i,
\qquad
r_j = \sqrt{\rho}\,r + \sqrt{1-\rho}\,\eta_j.
$$

Then

$$
x_i = \alpha P_i^{(R_{ij})} r_i + \varepsilon_i,
$$
$$
x_j = \alpha P_j^{(R_{ij})} r_j + \varepsilon_j,
$$
$$
x_k = \varepsilon_k \quad \text{for } k \notin \{i,j\}.
$$

The overlap parameter $\rho$ controls the strength of shared structure between the two active views.

### 1.8 Triple Redundancy Atom

For $R_{123}$, sample

$$
r,\eta_1,\eta_2,\eta_3 \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,I_m),
$$

and set

$$
r_k = \sqrt{\rho}\,r + \sqrt{1-\rho}\,\eta_k,\qquad k\in\{1,2,3\}.
$$

Then

$$
x_k = \alpha P_k^{(R_{123})} r_k + \varepsilon_k,\qquad k\in\{1,2,3\}.
$$

This induces shared structure across all three views.

### 1.9 Directional Synergy Atoms

For $S_{ij\to k}$, sample

$$
a,b \sim \mathcal{N}(0,I_m), \qquad h \in \mathcal{H}.
$$

Source views receive linearly projected latents:

$$
x_i = \alpha P_i^{(A_{ij})} a + \varepsilon_i,\qquad
x_j = \alpha P_j^{(B_{ij})} b + \varepsilon_j.
$$

The target view is generated from a fixed nonlinear readout network $\phi_h$:

$$
s_0 = \phi_h([a,b]) \in \mathbb{R}^m,
$$

followed by de-leakage via precomputed linear maps $C_a^{(h)}, C_b^{(h)}$:

$$
s = s_0 - C_a^{(h)} a - C_b^{(h)} b.
$$

Finally,

$$
x_k = \alpha P_k^{(\mathrm{SYN}_{ij})} s + \varepsilon_k.
$$

The de-leakage step suppresses single-source linear predictability in the target latent $s$, making the synergy structure more directional and harder to explain with single-view probes.

### 1.10 Synergy De-leakage Fitting (Offline, Per Dataset Seed)

For each hop $h$, de-leakage maps are fit from synthetic latent samples by least squares:

$$
W^{(h)} = \arg\min_W \| S_0 - XW \|_F^2 + \lambda \|W\|_F^2,
$$

where

$$
X = [A \; B] \in \mathbb{R}^{N \times 2m},\qquad S_0 \in \mathbb{R}^{N \times m}.
$$

The solution is partitioned as

$$
W^{(h)} =
\begin{bmatrix}
C_a^{(h)} \\
C_b^{(h)}
\end{bmatrix},
$$

which yields the de-leakage correction used during sample generation.

## 2. Validation Metrics (Raw Data, Before Any SSL Model)

### 2.1 Symmetric Dependence Proxy

Given two view matrices $X_A, X_B$ (rows are samples), define:

$$
D(X_A, X_B) = \frac{1}{2}\left(R^2(X_A \to X_B) + R^2(X_B \to X_A)\right),
$$

where $R^2$ is computed using ridge regression on a train/test split.

Expected U/R signatures:

- `U1/U2/U3`: low pairwise dependence,
- `Rij`: high dependence only for the matching pair,
- `R123`: elevated dependence for all pairs,
- dependence increases with $\rho$,
- dependence decreases with larger $\sigma$.

### 2.2 PCA-Based Intuition View

For a fixed atom, compute principal-component scores for each view:

$$
z_k = \mathrm{PC1}(X_k), \quad k\in\{1,2,3\}.
$$

Scatter plots of $(z_i,z_j)$ reveal the dominant pairwise geometric structure:

- diffuse clouds for unique atoms,
- aligned/elongated manifolds for redundant atoms.

This diagnostic is qualitative but highly interpretable and is useful for presentations and sanity checks.

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

The U/R-only subset corresponds to:

$$
\{0,1,2,3,4,5,6\} = \{U_1,U_2,U_3,R_{12},R_{13},R_{23},R_{123}\}.
$$

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
    test_plot_ur_compact_signature_grid_over_sigma,
    test_plot_ur_hyperparameter_sweeps_compact,
    test_plot_ur_intuition_scatter_examples,
)

test_plot_ur_compact_signature_grid_over_sigma()
test_plot_ur_hyperparameter_sweeps_compact()
test_plot_ur_intuition_scatter_examples()
print("Saved plots under test_outputs/pid_sar3")
PY
```

Output figures:

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

### 5.1 Figure A: U/R Signature Grid Across Noise

File:

- `test_outputs/pid_sar3/ur_compact_signature_grid_over_sigma.png`

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

### 5.2 Figure B: Hyperparameter Sweeps (`rho`, `sigma`, `alpha`)

File:

- `test_outputs/pid_sar3/ur_hyperparameter_sweeps_compact.png`

Left panel (redundancy overlap):

- For `R12`, the dependence proxy $D(x_1,x_2)$ should increase with $\rho$.
- Multiple curves at different `sigma` show that noise weakens observed dependence but preserves the monotonic trend.

Right panel (signal amplitude and noise):

- Mean $\|x_1\|_2$ increases as `alpha` increases.
- Larger `sigma` shifts norms upward and broadens the effective scale (signal + noise).
- Comparing `U1` and `R123` shows the effect under two different atom structures.

Why this is useful:

- It ties abstract hyperparameters to direct geometric/statistical effects in the observed data.

### 5.3 Figure C: PCA Intuition Scatter Plots (What You Liked)

File:

- `test_outputs/pid_sar3/ur_intuition_scatter_examples.png`

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
