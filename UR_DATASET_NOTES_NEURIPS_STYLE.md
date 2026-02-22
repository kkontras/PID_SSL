# U/R Subset: Intuition, Equations, and Reproducible Commands

This note documents the **Unique (U)** and **Redundancy (R)** subset of the `PID-SAR-3++` synthetic dataset, with:

- a compact mathematical description suitable for a NeurIPS-style methods section,
- interpretation of the PCA-based intuition plots,
- runnable commands to generate data and reproduce the U/R diagnostics.

The implementation lives in `pid_sar3_dataset.py`, and the plotting tests live in `tests/test_pid_sar3_dataset.py`.

## 1. Problem Setup (U/R-only subset)

We consider three observed views

\[
x_1, x_2, x_3 \in \mathbb{R}^d,
\]

and a latent dimensionality \(m \ll d\). Each sample contains **exactly one information atom** from the subset

\[
\mathcal{A}_{UR} = \{U_1,U_2,U_3,R_{12},R_{13},R_{23},R_{123}\}.
\]

The goal of this subset is to isolate:

- **Unique information**: signal present in exactly one view,
- **Redundant information**: shared signal across a pair or all three views.

This is the simplest regime for validating whether dependence metrics and encoders behave sensibly before introducing synergy.

## 2. Generative Model (Formal)

### 2.1 Fixed projection matrices

For each view \(k \in \{1,2,3\}\) and component \(c\), sample a fixed projection matrix

\[
P_k^{(c)} \in \mathbb{R}^{d \times m}, \qquad
P_k^{(c)}[:,\ell] \sim \mathcal{N}(0, I_d/d),
\]

followed by column normalization:

\[
P_k^{(c)}[:,\ell] \leftarrow \frac{P_k^{(c)}[:,\ell]}{\|P_k^{(c)}[:,\ell]\|_2}.
\]

These matrices are sampled once per dataset seed and then held fixed.

### 2.2 Global random variables and hyperparameters

Per sample, draw:

\[
\alpha \sim \mathrm{Uniform}(\alpha_{\min}, \alpha_{\max}),
\qquad
\varepsilon_k \sim \mathcal{N}(0, \sigma^2 I_d), \ \ k=1,2,3.
\]

For redundancy atoms, also draw an overlap parameter

\[
\rho \in \{0.2,0.5,0.8\}
\]

(or any user-defined set).

### 2.3 Unique atom \(U_i\)

Let \(u \sim \mathcal{N}(0,I_m)\). For atom \(U_i\),

\[
x_i = \alpha P_i^{(U_i)} u + \varepsilon_i,
\]
\[
x_j = \varepsilon_j \quad \text{for } j \neq i.
\]

Interpretation: only one view contains structured signal; the others are noise-only.

### 2.4 Pairwise redundancy atom \(R_{ij}\)

Let \(r,\eta_i,\eta_j \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,I_m)\), and define noisy shared latents

\[
r_i = \sqrt{\rho}\, r + \sqrt{1-\rho}\, \eta_i,
\qquad
r_j = \sqrt{\rho}\, r + \sqrt{1-\rho}\, \eta_j.
\]

Then

\[
x_i = \alpha P_i^{(R_{ij})} r_i + \varepsilon_i,
\qquad
x_j = \alpha P_j^{(R_{ij})} r_j + \varepsilon_j,
\]
\[
x_k = \varepsilon_k \quad \text{for } k \notin \{i,j\}.
\]

Interpretation: \(x_i\) and \(x_j\) share a common latent component whose strength increases with \(\rho\).

### 2.5 Triple redundancy atom \(R_{123}\)

Let \(r,\eta_1,\eta_2,\eta_3 \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,I_m)\), and

\[
r_k = \sqrt{\rho}\, r + \sqrt{1-\rho}\, \eta_k, \qquad k \in \{1,2,3\}.
\]

Then

\[
x_k = \alpha P_k^{(R_{123})} r_k + \varepsilon_k, \qquad k=1,2,3.
\]

Interpretation: all three views share the same latent factor (up to overlap/noise).

## 3. Compact Diagnostics for U/R (What the plots show)

The U/R diagnostics added in `tests/test_pid_sar3_dataset.py` are intentionally compact and intuitive.

### 3.1 Pairwise dependence proxy

For two views \(A,B\), define a symmetric dependence proxy using held-out linear prediction:

\[
D(A,B) = \frac{1}{2}\left(R^2(A \rightarrow B) + R^2(B \rightarrow A)\right),
\]

where each \(R^2\) is computed via ridge regression on a train/test split.

Expected signatures:

- `U1/U2/U3`: all pairwise \(D\) values are near zero (noise floor),
- `R12`: \(D(x_1,x_2)\) is highest,
- `R13`: \(D(x_1,x_3)\) is highest,
- `R23`: \(D(x_2,x_3)\) is highest,
- `R123`: all pairwise \(D\) values are elevated.

### 3.2 PCA intuition plots (PC1 vs PC1)

For a fixed atom (e.g., `U1`, `R12`, `R123`), collect many samples and compute the first principal component scores per view:

\[
z_k = \mathrm{PC1}(X_k), \quad X_k \in \mathbb{R}^{N \times d}.
\]

We then visualize scatter plots of \((z_1, z_2)\) for the same datapoints.

Interpretation:

- `U1`: diffuse cloud, weak pairwise structure between \(x_1\) and \(x_2\),
- `R12`: clear elongated alignment in \((\mathrm{PC1}(x_1), \mathrm{PC1}(x_2))\),
- `R123`: also aligned, since both views contain a shared latent factor.

Notes:

- PCA component sign is arbitrary, so correlation may flip sign across runs; magnitude is the meaningful quantity.
- Increasing `sigma` broadens the clouds and weakens visible alignment.

## 4. Hyperparameters and Their Visual Effect

The U/R compact plots are designed to explain the role of key hyperparameters:

- `sigma` (observation noise):
  - larger `sigma` lowers pairwise dependence and blurs PCA alignment.
- `rho` (redundancy overlap):
  - larger `rho` increases dependence in `Rij` and `R123`.
- `alpha` (signal amplitude):
  - larger `alpha` increases observed vector norm and improves separability from noise.

These appear directly in:

- `ur_compact_signature_grid_over_sigma.png`
- `ur_hyperparameter_sweeps_compact.png`
- `ur_intuition_scatter_examples.png`

## 5. Reproducible Commands

### 5.1 Generate the U/R intuition plots (recommended)

Run only the new U/R plot tests:

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

Or run the full plotting suite:

```bash
python tests/test_pid_sar3_dataset.py
```

### 5.2 Generate a U/R-only dataset and save to disk (`.npz`)

This command creates a balanced U/R training split and writes it to `data/`.

```bash
mkdir -p data
python - <<'PY'
import numpy as np
from pid_sar3_dataset import PIDDatasetConfig, PIDSar3DatasetGenerator

cfg = PIDDatasetConfig(
    d=32,
    m=8,
    sigma=0.45,
    alpha_min=0.8,
    alpha_max=1.2,
    rho_choices=(0.2, 0.5, 0.8),
    seed=0,
)
gen = PIDSar3DatasetGenerator(cfg)

# U/R-only pid_ids: U1,U2,U3,R12,R13,R23,R123
ur_pid_ids = [0, 1, 2, 3, 4, 5, 6]
n_per_atom = 5000
pid_schedule = np.repeat(ur_pid_ids, n_per_atom)

batch = gen.generate(n=len(pid_schedule), pid_ids=pid_schedule.tolist())
np.savez_compressed("data/pid_sar3_ur_train.npz", **batch)
print("Saved:", "data/pid_sar3_ur_train.npz", "with", len(pid_schedule), "samples")
PY
```

### 5.3 Generate train/val/test splits

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

make_split("data/pid_sar3_ur_train.npz", n_per_atom=10000)
make_split("data/pid_sar3_ur_val.npz",   n_per_atom=1000)
make_split("data/pid_sar3_ur_test.npz",  n_per_atom=1000)
PY
```

## 6. Suggested NeurIPS-Style Methods Paragraph (U/R subset)

We construct a synthetic three-view dataset \(x_1,x_2,x_3 \in \mathbb{R}^d\) in which each sample instantiates exactly one information atom from \(\{U_1,U_2,U_3,R_{12},R_{13},R_{23},R_{123}\}\). For each view and atom, we sample a fixed projection matrix \(P_k^{(c)} \in \mathbb{R}^{d \times m}\) at dataset initialization and keep it fixed throughout generation. Unique atoms are produced by projecting a latent Gaussian vector into a single view and adding isotropic Gaussian observation noise to all views. Redundancy atoms are produced by mixing a shared Gaussian latent with view-specific Gaussian perturbations via an overlap coefficient \(\rho\), then projecting the resulting latents into the corresponding views. Signal amplitude is modulated by a per-sample scalar \(\alpha\). This construction yields controlled pairwise and triple-view dependence patterns while preserving stochastic variability through \(\alpha\), \(\rho\), and \(\sigma\), enabling direct validation of representation objectives against known latent structure.

## 7. Suggested Results Text (PCA intuition)

On the U/R subset, PCA visualizations of \((\mathrm{PC1}(x_1), \mathrm{PC1}(x_2))\) provide an immediate qualitative check of the generator. Samples from `U1` form diffuse clouds with weak cross-view structure, consistent with signal being present only in view 1. In contrast, `R12` produces a visibly aligned manifold in the \((x_1,x_2)\) PCA plane, and `R123` also exhibits alignment because both views inherit a shared latent component. As expected, increasing observation noise `sigma` reduces alignment and compresses dependence-proxy values, while increasing redundancy overlap `rho` strengthens pairwise dependence in redundant atoms.

