# PID-SAR-3++: Complete Experimental Plan

## Single-Relationship Rotated PID Dataset for Evaluating M\>2 SSL Objectives

------------------------------------------------------------------------

# 0. Objective

We generate a synthetic 3-view dataset where **each datapoint contains
exactly one PID atom**: - Unique - Redundancy (pairwise or triple) -
Directional Synergy 

The goal is to evaluate how well different multi-view (M\>2) SSL
objectives preserve: - Unique information - Redundant information -
Directional synergistic information

Evaluation is done via controlled dependence and synergy-proxy metrics
on learned representations.

------------------------------------------------------------------------

# 1. Data Generation Plan

## 1.1 Output per sample

Each sample returns:

-   `x1, x2, x3 ∈ R^d`
-   `pid_id ∈ {0..9}`
-   `alpha` (signal amplitude)
-   `sigma` (noise scale)
-   `rho` (redundancy overlap; else -1)
-   `hop` (synergy depth; else 0)

### pid_id mapping

  ID   Relationship
  ---- --------------
  0    U1
  1    U2
  2    U3
  3    R12
  4    R13
  5    R23
  6    R123
  7    S12→3
  8    S13→2
  9    S23→1

------------------------------------------------------------------------

## 1.2 Global Parameters

Recommended defaults:

-   d = 32
-   m = 8
-   Train: 200k
-   Val: 20k
-   Test: 20k
-   alpha \~ Uniform(0.8, 1.2)
-   sigma = 0.5
-   rho ∈ {0.2, 0.5, 0.8}
-   hop ∈ {1,2,3,4}

------------------------------------------------------------------------

## 1.3 Fixed Objects (Sampled Once Per Dataset Seed)

### Projection Matrices

For each modality k and component:

P\[k\]\[comp\] ∈ R\^{d×m}, entries \~ N(0, 1/d)

Column-normalize each P matrix.

Components required: - U1,U2,U3 - R12,R13,R23,R123 -
A12,B12,A13,B13,A23,B23 - SYN12,SYN13,SYN23

------------------------------------------------------------------------

## 1.4 Synergy Readout Networks

Define fixed MLPs φ_h for h ∈ {1..Hmax}:

Input: concat(a,b) ∈ R\^{2m}\
Output: s0 ∈ R\^m

Depth = h residual blocks.

------------------------------------------------------------------------

## 1.5 Synergy De-leakage

Precompute linear maps C_a\[h\], C_b\[h\]:

For many sampled (a,b): s0 = φ_h(\[a,b\]) Fit least squares: s ≈ s0 -
C_a a - C_b b

At generation time: s = s0 - C_a\[h\] a - C_b\[h\] b

This suppresses single-view leakage.

------------------------------------------------------------------------

## 1.6 Per-Sample Generation

Initialize x1=x2=x3=0.

### Unique Ui

u \~ N(0,I) xi = alpha \* P\[i\]\['Ui'\] @ u

Add noise.

------------------------------------------------------------------------

### Redundancy Rij (Noisy Overlap)

r \~ N(0,I) eta_i, eta_j \~ N(0,I)

r_i = sqrt(rho)*r + sqrt(1-rho)*eta_i r_j = sqrt(rho)*r +
sqrt(1-rho)*eta_j

xi = alpha \* P\[i\]\['Rij'\] @ r_i xj = alpha \* P\[j\]\['Rij'\] @ r_j

Add noise.

------------------------------------------------------------------------

### Triple Redundancy

rk = sqrt(rho)*r + sqrt(1-rho)*eta_k

xk = alpha \* P\[k\]\['R123'\] @ rk

Add noise.

------------------------------------------------------------------------

### Directional Synergy Sij→k

a,b \~ N(0,I) hop = h

xi = alpha \* P\[i\]\['Aij'\] @ a xj = alpha \* P\[j\]\['Bij'\] @ b

s0 = φ_h(\[a,b\]) s = s0 - C_a\[h\] a - C_b\[h\] b

xk = alpha \* P\[k\]\['SYNij'\] @ s

Add noise.

------------------------------------------------------------------------

# 2. Generator Validation

## 2.1 Signature Checks (Raw X)

### Dependence Proxy D(A,B)

Linear regression R² or CCA.

Expected: - Ui: near-zero cross-view dependence - Rij: D(i,j) high;
others low - R123: all high - Redundancy monotonic in rho

------------------------------------------------------------------------

### Synergy Proxy

Train predictors: - xk ← xi - xk ← xj - xk ← \[xi,xj\]

Δ = R²_joint − max(R²_single)

Expected: - Δ \> 0 only for correct synergy direction - Δ decreases with
hop - Single-source R² ≈ noise floor

------------------------------------------------------------------------

# 3. Model Plan

## 3.1 Encoders

Three independent encoders:

d → 4p → 4p → p\
p = 64 recommended

------------------------------------------------------------------------

## 3.2 Projectors

p → 2p → q\
q = 64

Normalize for contrastive losses.

------------------------------------------------------------------------

## 4. SSL Objectives

### (A) Pairwise InfoNCE

Sum across (1,2), (1,3), (2,3)

### (B) Multi-positive InfoNCE

Anchor vs both other views simultaneously

### (C) VICReg-3

Pairwise invariance + variance + covariance

### (D) Joint Predictive Baseline

Predict one representation from the other two

------------------------------------------------------------------------

# 5. Training Setup

-   Batch size: 512
-   Steps: 100k
-   Optimizer: AdamW
-   LR: 1e-3 cosine decay
-   Same architecture across objectives
-   No use of pid_id during training

------------------------------------------------------------------------

# 6. Representation-Level Evaluation

Freeze encoders. Compute Zk.

## 6.1 Redundancy / Unique Proxy

D(Zi,Zj) via linear R²

Expected signatures per pid_id.

------------------------------------------------------------------------

## 6.2 Synergy Proxy

Train probes:

Zk ← Zi\
Zk ← Zj\
Zk ← \[Zi,Zj\]

Δ_ij→k = R²_joint − max(single)

Track Δ vs hop.

------------------------------------------------------------------------

# 7. Experimental Sweeps

## 7.1 Main Sweep

Objectives × hop × rho

Measure: - Synergy retention curves - Redundancy monotonicity - Grasp
vector per pid_id

------------------------------------------------------------------------

## 7.2 Weak View Robustness

Increase σ for one modality.

------------------------------------------------------------------------

## 7.3 Capacity Sweep

p ∈ {16,32,64}

------------------------------------------------------------------------

# 8. Acceptance Criteria

Before SSL: - Synergy Δ positive only in correct direction - Δ decreases
with hop - Redundancy D increases with rho

After SSL: - Objectives differ in grasp vectors - Predictive baseline
strongest on synergy - Global objectives favor R123 over Rij

------------------------------------------------------------------------

# End of Plan
