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

### 6.1.2 Repeated-Seed Secondary Diagnostics Snapshot (Quick CPU Run, `n=3`)

We ran the new repeated-seed summary harness (`test_main_results_four_models_repeated_seed_summary`) on February 23, 2026 in a short CPU regime (`3` dataset worlds / optimization seeds, `140` SSL steps). This is not the final benchmark, but it is enough to replace single-seed claims with uncertainty-aware summaries.

Secondary diagnostics (mean [95% CI]):

| Model | Family-3 acc | mean `R` recall | mean `R -> S` leakage | mean matched `R/S` centroid cos |
| --- | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.566 [0.552, 0.579] | 0.546 [0.520, 0.573] | 0.201 [0.183, 0.220] | 0.769 [0.724, 0.813] |
| B: pairwise InfoNCE | 0.560 [0.558, 0.562] | 0.520 [0.509, 0.532] | 0.200 [0.179, 0.221] | 0.921 [0.902, 0.939] |
| C: TRIANGLE | 0.606 [0.581, 0.631] | 0.642 [0.627, 0.657] | 0.186 [0.154, 0.217] | 0.936 [0.912, 0.960] |
| D: ConFu | 0.557 [0.551, 0.562] | 0.508 [0.476, 0.539] | 0.202 [0.188, 0.217] | 0.928 [0.909, 0.948] |

What this changes in the interpretation:

- **TRIANGLE remains strongest in this short regime** on family classification and `R` recall, and it also has the lowest mean `R -> S` leakage among the four in this run.
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

### 6.2 Legacy Fused Classification Check (Context Only)

Models:

1. `A`: sum of 3 unimodal SimCLR losses (`x1`, `x2`, `x3` trained separately)
2. `B`: sum of 3 pairwise InfoNCE losses (`(x1,x2)`, `(x1,x3)`, `(x2,x3)`)

This two-model fused classification comparison is retained only as historical context for the protocol fix and for geometry/pathology intuition. It is not the primary ranking result.

Context artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/pid10_confusions_fused_frozen_two_models.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_two_models_task_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_two_models_geometry_summary.csv`

Takeaway (context only):

- pairwise InfoNCE improves some local cross-modal alignment behavior but increases matched `R/S` centroid overlap, which was one reason PID-label classification alone was not a good headline benchmark.

### 6.3 Legacy 4-Model Fused Classification Comparison (Context Only)

We compare four methods under the same fused frozen protocol:

1. `A`: 3x unimodal SimCLR
2. `B`: pairwise InfoNCE sum (pairwise SimCLR/NT-Xent)
3. `C`: TRIANGLE (area contrastive; closer to the paper's core similarity than the earlier proxy)
4. `D`: ConFu (trainable pair-fusion heads + fused-pair-to-third contrastive terms)

Related papers:

- TRIANGLE (Grassucci et al.): *A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity*
- ConFu (Koutoupis et al.): *The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment*

This four-model fused classification comparison is useful for sanity checking geometry pathologies, but it is not the main benchmark because it over-emphasizes PID-label classification and latent-proxy `R²` summaries.

Context artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/pid10_confusions_fused_frozen_four_models.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/geometry_pid_summary_four_models.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_four_models_task_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_four_models_geometry_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/subset_family_probe_heatmaps_four_models.png` (preferred representation-level readout)

Takeaway (context only):

- the 4-model fused comparison is best used to inspect failure modes (especially `Rij <-> Sij->k` overlap), while method ranking should come from the source->target downstream matrix in Section `6.8.1`.

### 6.4 Redundancy Classification Sanity Check (Protocol Fix Evidence)

Using the corrected same-world split, redundancy terms are much more learnable than in the earlier cross-seed experiments.

This section is retained only to document why the corrected same-world split matters. The main-results document uses redundancy recall / `R->S` leakage as *secondary diagnostics* (see the repeated-seed snapshot in `6.1.2`), not as the headline benchmark.

Short conclusion:

- after fixing the split, redundancy terms become materially more learnable, confirming that earlier failures were partly protocol-induced rather than purely objective-induced.


### 6.8 Tuned Long-Run Downstream Benchmark (Primary Results)

This section is the primary SSL benchmark result in this document. The ranking target is the **source->target modality prediction matrix** in Section `6.8.1`, not PID-label classification and not latent-proxy `R²` tables.

Tuning protocol:

- same corrected same-world split
- `probe_train` for regressor fit
- `probe_val` for model/hyperparameter selection
- `probe_test` for final report
- selection metric used in the original tuning run: validation mean `R²` over latent proxy tasks (`y_macro_r2`)

Methods:

1. `A`: 3x unimodal SimCLR
2. `B`: pairwise InfoNCE
3. `C`: TRIANGLE (area contrastive)
4. `D`: ConFu
5. `E`: directional predictive hybrid (`[h_i,h_j] -> h_k`)

Supplementary latent-proxy (`y_*`) regression artifacts are kept for interpretability and tuning provenance, but they are secondary and are discussed in `docs/PID_SAR3_SSL_APPENDIX_ABLATIONS.md`.

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

Primary result for this section:

- **Figure 14 + Table 7b (the full all-source->target matrix)** are the main benchmark result.
- **Table 7** is a focused excerpt of the three rotated pair->target tasks (`23->1`, `13->2`, `12->3`) and should be read as a slice of `7b`, not as a separate benchmark.

Before the full matrix, a compact grouped summary helps orient the reader.

#### Table 7a. Grouped Summary Of The All Source->Target Matrix (macro-F1 averages over task groups; A-D only)

Source: `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1.csv`

| Model | self `1->1/2->2/3->3` | single cross-modal (`1->2`, etc.) | pair->heldout target (`23->1`, `13->2`, `12->3`) | pair->member target (`12->1`, etc.) | `123->target` |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.987 | 0.511 | 0.516 | 0.984 | 0.981 |
| B: pairwise InfoNCE | 0.986 | 0.614 | 0.639 | 0.982 | 0.978 |
| C: TRIANGLE | 0.986 | 0.615 | 0.651 | 0.982 | 0.979 |
| D: ConFu | 0.986 | 0.600 | 0.629 | 0.982 | 0.979 |

Interpretation of Table 7a:

- Self-prediction and overcomplete settings (`pair->member`, `123->target`) are near-ceiling for all methods, so they are sanity checks, not ranking metrics.
- The ranking signal lives in the **cross-modal** groups, especially **pair->heldout target**.
- `C: TRIANGLE` is strongest on the grouped pair->heldout target average; `B: pairwise InfoNCE` is close; `D: ConFu` remains competitive; `A` is near the cross-modal floor.

![Rotated pair->target downstream summary](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_summary.png)

*Figure 12. Main downstream benchmark: rotated pair->target modality classification with frozen encoders. Left: rotation-averaged `F1-skill`. Middle: heuristic “applicable PID” average. Right: applicability gap (`applicable - non-applicable`).*

![PID-by-rotation pair->target heatmaps](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_pid_rotation_heatmaps.png)

*Figure 13. Full `PID x rotation` pair->target downstream scores (`F1-skill`; random ≈ 0) for each method.*

![All source->target macro-F1 heatmaps](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1_heatmaps.png)

*Figure 14. Main downstream result: all source->target rotations (`1/2/3/12/13/23/123 -> 1/2/3`) reported as macro-F1 for A-D (frozen encoders). Includes self-prediction rows such as `1->1` and cross-modal rows such as `2->1`.*

#### Table 7. Focused Excerpt From The Main Matrix: Rotated Pair->Target Downstream Results (frozen encoders, held-out test)

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

What this clarifies (for the pair->target slice):

- This benchmark is much closer to the intended multimodal question than PID-label classification or latent `y_*` probes alone.
- **Cross-modal methods now clearly outperform unimodal SimCLR** on the true pair->target task (A is near-random on the normalized scale).
- **TRIANGLE is the strongest method across two of the three rotations** (`13->2`, `12->3`) when reporting macro-F1 directly.
- **ConFu** is competitive and strongest on one rotation (`23->1`).
- **Pairwise InfoNCE** is consistently strong and clearly above unimodal SimCLR on all three rotations.

#### Table 7b. Main Result Matrix: All Source->Target Rotations (macro-F1, A-D only)

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

Reading guide (how to use the main matrix):

- `1->1`, `2->2`, `3->3` are self-prediction sanity checks.
- `2->1`, `3->1`, `1->2`, ... are single-modality cross-modal transfers.
- `23->1`, `13->2`, `12->3` are the main rotated pair->target tasks summarized in Table 7.

Why `7b` should be treated as the main result (not just the rotated subset):

- It shows the **full cross-modal behavior surface**, not only three selected tasks.
- It separates ceiling sanity checks (`1->1`, `12->1`, `123->1`, etc.) from the tasks that actually rank methods.
- It makes it harder to overfit the narrative to a small subset of rotations.

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
