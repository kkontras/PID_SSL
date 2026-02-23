# PID-SAR-3++ SSL Main Results

This document is the primary SSL benchmark report for PID-SAR-3++. It contains the corrected evaluation protocol, the core comparison sequence, and the downstream pair->target benchmark used for method ranking; tuning sweeps and supporting diagnostics are moved to `docs/PID_SAR3_SSL_APPENDIX_ABLATIONS.md`.

## 6. SSL Results (Main Results)

The section numbering below is preserved from the original SSL report to keep table/figure references stable.

### 6.1 Evaluation Protocol (Important Correction)

Earlier SSL comparisons used different dataset seeds for train/test probe generators. In this dataset, the seed changes:

- fixed projection matrices
- fixed synergy MLP
- de-leakage maps

So cross-seed probing unintentionally tested transfer across different observation dictionaries, not just generalization to new samples.

Corrected protocol used here:

- same dataset seed for SSL training and probe splits
- different sampled examples for train/test
- frozen encoders
- concatenate `[h1,h2,h3]`
- linear probes on held-out data

Implementation:

- `tests/test_pid_sar3_ssl_fused_confusions.py`

### 6.1.1 Main-Results Reporting Contract (New)

To reduce “leaderboard dump” behavior, the main results are now defined as a small set of decision metrics reported with uncertainty across repeated runs, not a single seed snapshot.

Main-results requirements:

- report repeated runs across dataset worlds and optimization seeds (not one run)
- report mean and 95% CI for the primary metrics
- keep one primary ranking target and a small number of failure-mode metrics
- move tuning sweeps and broad diagnostic tables to `docs/PID_SAR3_SSL_APPENDIX_ABLATIONS.md`

Primary decision metrics (current fused comparison stage):

- downstream task score for the declared primary benchmark (when available in the active harness)
- redundancy recall (`R` rows in PID confusion)
- `R -> S` leakage from PID confusions
- matched `R/S` centroid overlap (geometry pathology metric)

Implementation artifact (repeated-seed summary):

- `test_outputs/pid_sar3_ssl_fused_confusions/main_results_four_models_seeded_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/main_results_four_models_seeded_trials.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/main_results_four_models_seeded_summary.png`

### 6.1.2 Repeated-Seed Snapshot (Quick CPU Run, `n=3`)

We ran the new repeated-seed summary harness (`test_main_results_four_models_repeated_seed_summary`) on February 23, 2026 in a short CPU regime (`3` dataset worlds / optimization seeds, `140` SSL steps). This is not the final benchmark, but it is enough to replace single-seed claims with uncertainty-aware summaries.

Primary metrics (mean [95% CI]):

| Model | PID-10 acc | Family-3 acc | mean `R` recall | mean `R -> S` leakage | mean matched `R/S` centroid cos |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.569 [0.555, 0.584] | 0.566 [0.552, 0.579] | 0.546 [0.520, 0.573] | 0.201 [0.183, 0.220] | 0.769 [0.724, 0.813] |
| B: pairwise InfoNCE | 0.527 [0.521, 0.533] | 0.560 [0.558, 0.562] | 0.520 [0.509, 0.532] | 0.200 [0.179, 0.221] | 0.921 [0.902, 0.939] |
| C: TRIANGLE | 0.657 [0.647, 0.668] | 0.606 [0.581, 0.631] | 0.642 [0.627, 0.657] | 0.186 [0.154, 0.217] | 0.936 [0.912, 0.960] |
| D: ConFu | 0.529 [0.512, 0.547] | 0.557 [0.551, 0.562] | 0.508 [0.476, 0.539] | 0.202 [0.188, 0.217] | 0.928 [0.909, 0.948] |

What this changes in the interpretation:

- **TRIANGLE remains strongest in this short regime** on `PID-10` and `R` recall, and it also has the lowest mean `R -> S` leakage among the four in this run.
- **Unimodal SimCLR still has much lower matched `R/S` centroid overlap** than the cross-modal methods (better geometry on this pathology metric), so the story is not a single scalar ranking.
- **Pairwise InfoNCE and ConFu are close on several summary metrics** in this short regime, which is exactly the kind of claim that should be reported with intervals rather than one-run tables.

Short-run caveat:

- The supplementary synergy proxy `R²(y_s12_3)` remains highly unstable and strongly negative in this configuration (large variance across the `n=3` runs), so it should stay out of headline ranking claims in the main text.

### 6.1.3 Lead Readout: Do The Encoders Capture U / R / S?

Before showing PID-10 confusions, the main results should lead with a subset-based family classification probe on frozen features:

- subsets: `x1`, `x2`, `x3`, `x12`, `x13`, `x23`, `x123`
- metrics: `Family-3` accuracy, `Family-3` macro-F1, and one-vs-rest `F1` for `U`, `R`, `S`
- purpose: answer the direct representation question first ("what does each subset encode about unique/redundancy/synergy families?")

Artifacts (4-model fused comparison):

- `test_outputs/pid_sar3_ssl_fused_confusions/subset_family_probe_heatmaps_four_models.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_four_models_subset_predictors.csv`

This readout should be presented before PID-10 leaderboards because it is a cleaner representation-level diagnostic and makes the U/R/S tradeoffs visible without collapsing them into one class label metric.

Quick snapshot from the current 4-model fused run (`x123` subset only; full subset grid is in the heatmap/CSV above):

| Model | Family-3 acc (`x123`) | Family-3 macro-F1 (`x123`) | `U` F1 (`x123`) | `R` F1 (`x123`) | `S` F1 (`x123`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.586 | 0.580 | 0.609 | 0.620 | 0.510 |
| B: pairwise InfoNCE | 0.567 | 0.560 | 0.608 | 0.602 | 0.469 |
| C: TRIANGLE | 0.619 | 0.611 | 0.636 | 0.664 | 0.533 |
| D: ConFu | 0.589 | 0.582 | 0.583 | 0.634 | 0.530 |

This table is intentionally not the full story. The full subset matrix (`x1`, `x2`, `x3`, `x12`, `x13`, `x23`, `x123`) is the point of the diagnostic, because it shows which methods improve specifically when more modalities are exposed.

### 6.2 Core Comparison (2 Models, Fused Frozen Encoders)

Models:

1. `A`: sum of 3 unimodal SimCLR losses (`x1`, `x2`, `x3` trained separately)
2. `B`: sum of 3 pairwise InfoNCE losses (`(x1,x2)`, `(x1,x3)`, `(x2,x3)`)

Primary figure:

![PID-10 confusion matrices, fused frozen encoders](test_outputs/pid_sar3_ssl_fused_confusions/pid10_confusions_fused_frozen_two_models.png)

*Figure 8. PID-10 confusion matrices under the corrected same-world split protocol (frozen encoders + concatenated modalities + linear probe).*

#### Table 3. Two-Model Fused Frozen Summary

Source: `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_two_models_task_summary.csv`

| Task | A: 3x unimodal SimCLR | B: pairwise InfoNCE | `B - A` |
| --- | ---: | ---: | ---: |
| `PID-10` accuracy | 0.596 | 0.527 | -0.069 |
| `Family-3` accuracy | 0.581 | 0.565 | -0.016 |
| `R²(y_u1)` | 0.496 | 0.505 | +0.009 |
| `R²(y_r12)` | 0.180 | 0.063 | -0.118 |
| `R²(y_r123)` | 0.265 | 0.236 | -0.029 |
| `R²(y_s12_3)` | -0.335 | -0.530 | -0.195 |

#### Table 4. Two-Model Geometry Summary (Why Confusions Differ)

Source: `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_two_models_geometry_summary.csv`

| Metric | A: 3x unimodal SimCLR | B: pairwise InfoNCE |
| --- | ---: | ---: |
| overall mean margin (PID classes) | -0.037 | -0.077 |
| mean margin on `R` classes | -0.053 | -0.125 |
| matched `R/S` centroid cosine (mean) | ~0.769 | ~0.941 |
| matched `R/S` nearest-centroid pair acc (mean) | ~0.589 | ~0.606 |

Interpretation:

- Pairwise InfoNCE preserves local matched-pair separability reasonably well.
- But it creates much stronger global overlap between matched redundancy/synergy class centroids (`Rij` and `Sij->k`), which hurts PID-10 class separation.

### 6.3 Higher-Order Alignment Comparison (4 Models)

We compare four methods under the same fused frozen protocol:

1. `A`: 3x unimodal SimCLR
2. `B`: pairwise InfoNCE sum (pairwise SimCLR/NT-Xent)
3. `C`: TRIANGLE (area contrastive; closer to the paper's core similarity than the earlier proxy)
4. `D`: ConFu (trainable pair-fusion heads + fused-pair-to-third contrastive terms)

Related papers:

- TRIANGLE (Grassucci et al.): *A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity*
- ConFu (Koutoupis et al.): *The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment*

Primary figures:

![PID-10 confusion matrices, 4-model comparison](test_outputs/pid_sar3_ssl_fused_confusions/pid10_confusions_fused_frozen_four_models.png)

*Figure 9. PID-10 confusion matrices for the 4-model comparison (fused frozen validation).*

![Geometry and PID summary, 4 models](test_outputs/pid_sar3_ssl_fused_confusions/geometry_pid_summary_four_models.png)

*Figure 10. Compact geometry/pathology summary across the 4 models. Left: matched `R/S` centroid overlap (lower better). Middle: matched-pair separability (higher better). Right: PID-10 accuracy.*

#### Table 5. Key 4-Model Results (Fused Frozen Encoders)

Sources:

- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_four_models_task_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_four_models_geometry_summary.csv`

| Model | PID-10 | Family-3 | mean matched `R/S` centroid cosine | mean matched `R/S` NC acc | `R²(y_r12)` | `R²(y_r123)` | `R²(y_s12_3)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.580 | 0.590 | 0.729 | 0.582 | 0.398 | 0.430 | -0.252 |
| B: pairwise InfoNCE | 0.555 | 0.567 | 0.911 | 0.604 | 0.103 | 0.365 | -0.635 |
| C: TRIANGLE | 0.658 | 0.619 | 0.947 | 0.614 | 0.389 | 0.387 | -0.677 |
| D: ConFu | 0.548 | 0.589 | 0.924 | 0.547 | 0.273 | 0.475 | -0.459 |

What matters:

- **TRIANGLE** is best on `PID-10` / `Family-3` in this regime.
- **ConFu** is strongest on `R²(y_r123)` and `R²(y_s12_3)` among the four.
- **Unimodal SimCLR** remains surprisingly strong and is best on `R²(y_r12)` in this run.
- All methods still exhibit the `Rij <-> Sij->k` confusion pathology.

### 6.4 Redundancy-Focused Classification (Sanity Check That Prompted the Split Fix)

Using the corrected same-world split, redundancy terms are much more learnable than in the earlier cross-seed experiments.

From PID confusion matrices (rows `R12/R13/R23/R123`):

- TRIANGLE: avg `R` recall `0.686`, `R -> S` leakage `0.136` (best)
- Unimodal SimCLR: avg `R` recall `0.575`
- Pairwise InfoNCE: avg `R` recall `0.542`
- ConFu: avg `R` recall `0.533`

This confirms the earlier poor shared-information results were largely caused by the split protocol, not only by the SSL objective choice.


### 6.8 Tuned Long-Run Downstream Proxy Benchmark (600 Steps, Frozen Encoders, `x123` Main Evaluation)

This is now the primary SSL benchmark result.

Instead of predicting the `PID-10` term label, we evaluate pretrained encoders by frozen-feature downstream regression on the latent proxy targets `y_*`. These are the intended PID-information probes.

We expanded the generator aux outputs to expose the full symmetric set of latent targets (10 total):

- unique: `y_u1`, `y_u2`, `y_u3`
- redundancy: `y_r12`, `y_r13`, `y_r23`, `y_r123`
- synergy: `y_s12_3`, `y_s13_2`, `y_s23_1`

Evaluation protocol (main experiment):

- train SSL encoder(s)
- freeze encoders
- concatenate all three modalities (`x123` -> `[h1,h2,h3]`)
- fit downstream linear regressors (Ridge) for each `y_*` target on the masked subsets
- report held-out `R²`

Tuning protocol:

- same corrected same-world split
- `probe_train` for regressor fit
- `probe_val` for model/hyperparameter selection
- `probe_test` for final report
- selection metric (for the base downstream-tuned models): validation mean `R²` over all 10 `y_*` tasks (`y_macro_r2`)

Methods:

1. `A`: 3x unimodal SimCLR
2. `B`: pairwise InfoNCE
3. `C`: TRIANGLE (area contrastive)
4. `D`: ConFu
5. `E`: directional predictive hybrid (`[h_i,h_j] -> h_k`)

Primary downstream artifacts (regression version, kept as supplementary):

- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_y_downstream_model_selection.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_y_downstream_selected_hparams.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_y_downstream_x123_task_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_y_downstream_x123_summary.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_y_downstream_subset_ablations.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_y_downstream_subset_ablations.png`

### 6.8.1 Rotated Pair->Target Modality Classification (Primary Metric)

The main downstream benchmark should use modalities directly:

- input: **two modalities**
- target: **the third modality**
- frozen encoders
- rotate across all three directions: `23 -> 1`, `13 -> 2`, `12 -> 3`

This avoids introducing a separate hand-designed target variable as the primary task. Instead, we test whether the pretrained representation supports actual cross-modal prediction.

Task construction (classification, random ≈ 0 baseline on the normalized score):

- use frozen features of the two input modalities (concatenated)
- predict the target modality observation vector `x_target`
- convert each target dimension into a binary classification task by thresholding at the **train median** (per dimension)
- fit a linear classifier per target dimension
- average across target dimensions

Reported metrics:

- `macro-F1` (averaged over target dimensions)
- `κ` (Cohen's kappa; random near `0`)
- `F1-skill = (F1 - 0.5) / 0.5` for the median-balanced binary tasks, so **random ≈ 0**

We evaluate:

- overall per rotation (`23->1`, `13->2`, `12->3`)
- per PID atom within each rotation (full `PID x rotation` table/heatmap)

Primary pair->target artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_overall_rotation_scores.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_pid_rotation_scores.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_summary.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_pid_rotation_heatmaps.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1_heatmaps.png`

![Rotated pair->target downstream summary](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_summary.png)

*Figure 12. Main downstream benchmark: rotated pair->target modality classification with frozen encoders. Left: rotation-averaged `F1-skill`. Middle: heuristic “applicable PID” average. Right: applicability gap (`applicable - non-applicable`).*

![PID-by-rotation pair->target heatmaps](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_pid_rotation_heatmaps.png)

*Figure 13. Full `PID x rotation` pair->target downstream scores (`F1-skill`; random ≈ 0) for each method.*

![All source->target macro-F1 heatmaps](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1_heatmaps.png)

*Figure 14. All source->target rotations (`1/2/3/12/13/23/123 -> 1/2/3`) reported as macro-F1 for A-D (frozen encoders). Includes self-prediction rows such as `1->1` and cross-modal rows such as `2->1`.*

#### Table 7. Rotated Pair->Target Downstream Results (frozen encoders, held-out test; primary report uses per-rotation macro-F1)

Sources:

- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_overall_rotation_scores.csv`

| Model | `23->1` macro-F1 | `13->2` macro-F1 | `12->3` macro-F1 |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.520 | 0.518 | 0.514 |
| B: pairwise InfoNCE | 0.628 | 0.649 | 0.640 |
| C: TRIANGLE | 0.643 | 0.655 | 0.653 |
| D: ConFu | 0.647 | 0.599 | 0.640 |

Rotation-level highlights:

- `23->1`: **D** is strongest on macro-F1 (`0.647`), with `C` close (`0.643`)
- `13->2`: **C** is strongest (`0.655`)
- `12->3`: **C** is strongest (`0.653`)

What this clarifies:

- This benchmark is much closer to the intended multimodal question than PID-label classification or latent `y_*` probes alone.
- **Cross-modal methods now clearly outperform unimodal SimCLR** on the true pair->target task (A is near-random on the normalized scale).
- **TRIANGLE is the strongest method across two of the three rotations** (`13->2`, `12->3`) when reporting macro-F1 directly.
- **ConFu** is competitive and strongest on one rotation (`23->1`).
- **Pairwise InfoNCE** is consistently strong and clearly above unimodal SimCLR on all three rotations.

#### Table 7b. All Source->Target Rotations (macro-F1, A-D only)

Source: `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1.csv`

| Source->Target | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| `1->1` | 0.988 | 0.985 | 0.986 | 0.986 |
| `2->1` | 0.513 | 0.610 | 0.609 | 0.592 |
| `3->1` | 0.508 | 0.601 | 0.613 | 0.627 |
| `12->1` | 0.984 | 0.982 | 0.981 | 0.983 |
| `13->1` | 0.984 | 0.981 | 0.982 | 0.982 |
| `23->1` | 0.515 | 0.628 | 0.643 | 0.647 |
| `123->1` | 0.981 | 0.978 | 0.978 | 0.979 |
| `1->2` | 0.513 | 0.612 | 0.611 | 0.581 |
| `2->2` | 0.987 | 0.986 | 0.986 | 0.986 |
| `3->2` | 0.512 | 0.632 | 0.621 | 0.585 |
| `12->2` | 0.984 | 0.982 | 0.982 | 0.982 |
| `13->2` | 0.516 | 0.649 | 0.655 | 0.599 |
| `23->2` | 0.984 | 0.982 | 0.982 | 0.982 |
| `123->2` | 0.981 | 0.979 | 0.979 | 0.979 |
| `1->3` | 0.507 | 0.608 | 0.619 | 0.623 |
| `2->3` | 0.510 | 0.620 | 0.616 | 0.590 |
| `3->3` | 0.987 | 0.985 | 0.986 | 0.986 |
| `12->3` | 0.517 | 0.640 | 0.653 | 0.640 |
| `13->3` | 0.984 | 0.981 | 0.983 | 0.983 |
| `23->3` | 0.984 | 0.982 | 0.982 | 0.983 |
| `123->3` | 0.981 | 0.978 | 0.980 | 0.980 |

Reading guide:

- `1->1`, `2->2`, `3->3` are self-prediction sanity checks.
- `2->1`, `3->1`, `1->2`, ... are single-modality cross-modal transfers.
- `23->1`, `13->2`, `12->3` are the main rotated pair->target tasks from Table 7.

Important note on the heuristic “applicable PID” averages:

- We also computed a simple heuristic split of PID atoms into “applicable / non-applicable” for each rotation, but the averages are noisy and not yet a reliable primary metric.
- The full `PID x rotation` heatmaps are more informative than the heuristic scalar summary.
- We also tested a directional predictive hybrid (`E`) in exploratory runs, but it is omitted from the primary table here per the current reporting preference.


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
