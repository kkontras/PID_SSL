# PID-SAR-3++ SSL Appendix: Ablations and Diagnostics

This appendix contains supporting SSL analyses that inform interpretation but are not the primary ranking evidence: subset-feature diagnostics, anomaly notes, hyperparameter sweeps, supplementary `y_*` proxy evaluations, and secondary ablations. The section numbering below is preserved from the original SSL report for cross-reference stability.

## 6. SSL Results (Appendix)

### 6.5 Subset Predictor Diagnostics (What 1/2/3 Modalities Explain)

We added frozen-feature linear probes over modality subsets (`x1`, `x2`, `x3`, `x12`, `x13`, `x23`, `x123`) to inspect what the encoders actually encode.

![Subset predictor heatmaps (4 models)](test_outputs/pid_sar3_ssl_fused_confusions/subset_predictor_heatmaps_four_models.png)

*Figure 11. Subset predictor diagnostics using 1-, 2-, and 3-modality frozen features for each model.*

Key patterns (from `fused_frozen_four_models_subset_predictors.csv`):

- `PID-10` accuracy improves strongly from 1 -> 2 -> 3 modalities for all models.
- TRIANGLE is strongest on fused `x123` PID-10 (`0.658`).
- ConFu-style (fusion-head) is strongest on fused `x123` for `y_r123` and `y_s12_3` in this run.

### 6.6 What Still Looks Weird (and Why That Can Happen)

Even after fixing the split, some methods still look closer than expected. Working hypotheses:

1. **Short training regime masks objective differences**
   - At `~140` steps, some methods may still be in the same optimization phase.
2. **Global PID-10 is too coarse by itself**
   - Objective-specific differences show up more clearly in `Rij <-> Sij->k` confusions, geometry, and latent probes.
3. **Hyperparameters are not tuned per method**
   - TRIANGLE and ConFu likely need different temperatures and term weights.
4. **Objectives have real tradeoffs**
   - Some improve class separation, others improve latent recoverability.

### 6.7 Hyperparameter Sensitivity and Tuning (What Actually Matters)

We ran a reduced-regime sweep (same-world split, smaller probe set, 120-step default) to test whether tuning matters.

Source: `test_outputs/pid_sar3_ssl_fused_confusions/hparam_sensitivity_compact.csv`

Sweeps:

- pairwise InfoNCE temperature: `0.1, 0.2, 0.4`
- TRIANGLE temperature: `0.1, 0.2, 0.4`
- ConFu fusion weight (`confu_fused_weight`): `0.25, 0.5, 0.75` with `confu_pair_weight = 1 - fused`
- TRIANGLE training steps: `80, 240` (vs default `120`)
- directional predictive hybrid: `directional_pred_weight ∈ {0.1, 0.25, 0.5, 1.0, 2.0}`, `temp ∈ {0.1,0.2,0.4}`, and `steps ∈ {80,240,400}` in a reduced sweep

#### Table 6. Hyperparameter Sweep Highlights (Reduced Regime)

| Method / Sweep | Best setting (for PID-10 in sweep) | PID-10 | What moved |
| --- | --- | ---: | --- |
| pairwise InfoNCE temp | `temp=0.2` | 0.539 | temperature changes class scores and latent probes materially |
| TRIANGLE temp | `temp=0.2` | 0.673 | strong sensitivity; clear best in this sweep |
| ConFu fusion weight | `fused=0.75` (PID/Family) | 0.542 | fused weight shifts classification vs latent-task tradeoff |
| TRIANGLE steps | `steps=240` | 0.705 | longer training improves TRIANGLE substantially |
| directional hybrid weight | `directional_pred_weight=0.1` (PID) / `2.0` (synergy target) | 0.544 / `R²(y_s12_3)=-0.231` | strong tradeoff between classification and directional target recoverability |
| directional hybrid steps | `steps=400` (in sweep) | 0.592 | training budget helps classification, but latent probes can move non-monotonically |

Main tuning conclusions:

- **Yes, hyperparameters matter and should be tuned per method.**
- **TRIANGLE is especially sensitive to temperature and training budget.**
- **ConFu-style is sensitive to pair-vs-fused weighting**, and different settings favor different targets.
- **Directional predictive hybrid is sensitive to `directional_pred_weight`**: larger weight helps `y_s12_3` recoverability but can hurt PID-10 classification.
- A single shared hyperparameter point is not sufficient for a fair comparison.
- The reduced sweep is useful for direction finding, but **final method ranking should use a longer run with explicit tuning selection**.

Additional directional sweep artifact:

- `test_outputs/pid_sar3_ssl_fused_confusions/directional_predictive_hparam_sweep_compact.csv`


### 6.8.2 Downstream Proxy Classification on `y_*` (Supplementary Diagnostic)

We convert each scalar `y_*` target into a balanced classification task by binning it into `5` quantile bins (fit on the train split only, per task), then train frozen-feature linear classifiers (multinomial logistic regression) on the masked subsets.

Reported metrics:

- `macro-F1`
- `κ` (Cohen's kappa; random baseline near `0`)
- `F1-skill = (macro-F1 - 1/K) / (1 - 1/K)` with `K=5`, so **random guessing is approximately `0`**

Supplementary `y_*` classification artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_x123_task_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_x123_summary.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_subset_ablations.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_subset_ablations.png`

![Tuned x123 downstream classification summary](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_x123_summary.png)

*Figure 15. Supplementary diagnostic: frozen `x123` downstream proxy classification on latent `y_*` targets. Left: per-target `F1-skill`. Right: family-level macro `F1-skill` summaries (random ≈ 0).*

#### Table 8. Tuned 600-Step `y_*` Downstream Classification Proxy Results (`x123`, held-out test)

Sources:

- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_x123_task_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_y_downstream_selected_hparams.csv` (hyperparameters reused from the downstream regression tuning run)

| Model | Selected hparams | all-`y` macro-F1 | all-`y` `F1-skill` | all-`y` `κ` | unique `F1-skill` | redundancy `F1-skill` | synergy `F1-skill` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | `temp=0.1` | **0.316** | **0.145** | **0.155** | **0.258** | **0.174** | -0.005 |
| B: pairwise InfoNCE | `temp=0.1` | 0.297 | 0.121 | 0.129 | 0.225 | 0.152 | -0.023 |
| C: TRIANGLE exact | `temp=0.2` | 0.308 | 0.135 | 0.143 | 0.201 | 0.172 | **0.020** |
| D: ConFu fusion-head | `temp=0.1`, pair/fused=`0.25/0.75` | 0.279 | 0.098 | 0.105 | 0.177 | 0.110 | 0.004 |
| E: directional predictive hybrid | `temp=0.4`, `directional_pred_weight=0.5` | 0.305 | 0.131 | 0.138 | 0.210 | 0.158 | 0.015 |

#### Table 9. Selected Per-Target `y_*` Classification Results (held-out test, frozen `x123`)

Source: `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_x123_task_summary.csv`

| Target | Best model (`F1-skill`) | Notes |
| --- | --- | --- |
| `y_u1`, `y_u2`, `y_u3` | A (all three) | strongest unique-factor class separability under frozen linear probes |
| `y_r12` | C (`0.182`) | TRIANGLE best on this pairwise redundancy target |
| `y_r13` | E (`0.183`) | directional hybrid edges out others on this redundancy target |
| `y_r23` | B (`0.261`) | pairwise InfoNCE strongest on this pairwise redundancy target |
| `y_r123` | C (`0.199`) | TRIANGLE strongest on the 3-way redundancy target in classification form |
| `y_s12_3` | A (`0.041`) | weak but positive; most others near/below zero |
| `y_s13_2` | E (`0.034`) | directional hybrid best here |
| `y_s23_1` | C (`0.049`) | TRIANGLE best here |

What this clarifies (and why this is a better primary benchmark than `PID-10`):

- **The SSL methods are not interchangeable once the downstream target is explicit.**
- **Ranking depends on the evaluation objective**:
  - TRIANGLE wins on PID-term classification (Section 6.3 / classification-style analyses),
  - but **unimodal SimCLR wins on frozen linear latent recoverability (`y_*` downstream probes)**.
- **All methods are above random on many downstream tasks**, and the `F1-skill` scale makes that easy to read (`0` ≈ random).
- **Unique and redundancy proxies are consistently learnable** (positive family `F1-skill` across methods), while **synergy remains much weaker**.
- **TRIANGLE** is not the best overall downstream method, but it improves some specific redundancy/synergy proxy classification tasks (e.g. `y_r12`, `y_r123`, `y_s23_1`).
- **Directional predictive hybrid** shows targeted gains on some synergy/redundancy proxies (`y_s13_2`, `y_r13`) but not enough to win the all-`y` downstream metric.

This is the right interpretation target for the benchmark: frozen encoders should be judged by what PID-related latent variables they make linearly accessible, not only by a supervised PID label classifier.

#### Secondary Ablation (Subsets of Modalities)

Subset ablations are now explicitly secondary and only used to diagnose what the encoders encode.

Artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_subset_ablations.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_subset_ablations.csv`

![Subset ablations on y-task classification families](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_ycls_subset_ablations.png)

*Figure 16. Secondary ablations: frozen-feature downstream `y_*` classification family summaries (`F1-skill`) for modality subsets (`x1`, `x2`, `x3`, `x12`, `x13`, `x23`, `x123`).*

Key ablation pattern:

- `x123` is best for the downstream classification proxy metric for A, B, D, and E.
- For C (TRIANGLE), `x23` slightly outperforms `x123` on all-`y` `F1-skill` in this run.
- This reinforces why subset ablations should be diagnostic only, not the main ranking criterion.

Supporting note:

- We keep both the regression-based downstream probes (`tuned_long_steps_600_y_downstream_*`) and the PID-label classification artifacts (`tuned_long_steps_600_five_models_*`) as supporting analyses, but the primary SSL conclusion in this notes file is now based on the **rotated pair->target modality** downstream benchmark (Section 6.8.1). The `y_*` probe suites remain useful diagnostics for interpretability.

### 6.9 What To Do Next (Downstream-First)

1. Add a **synergy-focused tuning track** (select on validation synergy `F1-skill` instead of all-`y`) and compare with the all-`y` selected models.
2. Evaluate both `h` and `z` (encoder vs projector outputs) for the downstream `y_*` classification probes; some objectives may hide more linear information in `h`.
3. Add regime-stratified downstream results (`rho`, `sigma`, `hop`) to identify where higher-order methods help or hurt.
4. Add formal `R-only` and `S-only` benchmark stages with the same frozen-encoder downstream protocol.
5. Keep PID-term classification as a separate supervised stress test, not the main SSL ranking metric.

### 6.10 Reproducing the SSL Comparisons

```bash
python - <<'PY'
from tests.test_pid_sar3_ssl_fused_confusions import test_plot_fused_confusions_two_models
from tests.test_pid_sar3_ssl_fused_confusions import test_plot_fused_confusions_four_models_higher_order

test_plot_fused_confusions_two_models()
test_plot_fused_confusions_four_models_higher_order()
print("Saved outputs under test_outputs/pid_sar3_ssl_fused_confusions")
PY
```
