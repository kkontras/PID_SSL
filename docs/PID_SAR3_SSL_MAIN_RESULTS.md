# PID-SAR-3++ SSL Main Results

This document reports the primary SSL results for PID-SAR-3++. It presents the corrected evaluation protocol, a compact set of representation-level diagnostics, and the source->target downstream benchmark used for method ranking. Hyperparameter sweeps, supplementary latent-proxy analyses, and broader ablations are reported separately in `docs/PID_SAR3_SSL_APPENDIX_ABLATIONS.md`.

## 6. SSL Results (Main Results)

The section numbering below is preserved from the original SSL report to keep table/figure references stable.

Main finding: on the regenerated 5-fold source->target evaluation reported with **Cohen's \(\kappa\)** (Section `6.8.1`), all methods are near chance on the hardest cross-modal pair->heldout-target tasks in this setup, while self/overcomplete tasks remain strong; the main practical conclusion is therefore that **the current training/evaluation regime is not yet producing robust cross-modal transfer under a chance-corrected metric**.

### 6.1 Evaluation Protocol (Important Correction)

Earlier comparisons used different dataset seeds for probe-train and probe-test generators. In PID-SAR-3++, changing the dataset seed changes the fixed projection operators, the fixed synergy network, and the de-leakage maps, so cross-seed probing evaluates transfer across different observation dictionaries rather than ordinary generalization to new samples from the same world.

All results reported here therefore use a same-world split: the SSL model and probe splits share the same dataset seed, while train/test probe examples are sampled independently. Encoders are frozen at evaluation time, the three modality embeddings are concatenated as `[h_1,h_2,h_3]`, and linear probes are fit on held-out data.

Implementation:

- `tests/test_pid_sar3_ssl_fused_confusions.py`

### 6.1.1 Main-Results Reporting Contract

To avoid presenting the benchmark as a single-seed leaderboard, the main results are defined as a small set of decision metrics reported with uncertainty across repeated runs. In practice, we report the sample mean \(\bar{x}\) together with the standard error \(\mathrm{SE}=s/\sqrt{n}\), where \(s\) is the sample standard deviation over runs and \(n\) is the number of runs.

The main text prioritizes one ranking target (the source->target matrix in Section `6.8.1`) together with a small number of failure-mode diagnostics, while broader sweeps and auxiliary tables are moved to the appendix.

### 6.1.2 Repeated-Seed Secondary Diagnostics Snapshot (Quick CPU Run, `n=3`)

We ran the repeated-seed summary harness (`test_main_results_four_models_repeated_seed_summary`) on February 23, 2026 in a short CPU regime (`n=3` dataset worlds / optimization seeds; `140` SSL steps). This run is not the final benchmark, but it is sufficient to replace single-seed statements with uncertainty-aware summaries.

Secondary diagnostics (mean \(\pm\) SE):

| Model | Family-3 acc | Family-3 \(\kappa\) | mean `R` recall | mean `R -> S` leakage | mean matched `R/S` centroid cos |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.566 ± 0.0068 | 0.342 ± 0.0080 | 0.546 ± 0.0135 | 0.201 ± 0.0094 | 0.769 ± 0.0227 |
| B: pairwise InfoNCE | 0.560 ± 0.0010 | 0.331 ± 0.0012 | 0.520 ± 0.0059 | 0.200 ± 0.0108 | 0.921 ± 0.0095 |
| C: TRIANGLE | 0.606 ± 0.0129 | 0.401 ± 0.0193 | 0.642 ± 0.0076 | 0.186 ± 0.0162 | 0.936 ± 0.0122 |
| D: ConFu | 0.557 ± 0.0028 | 0.326 ± 0.0030 | 0.508 ± 0.0161 | 0.202 ± 0.0072 | 0.928 ± 0.0101 |

Interpretation of the secondary diagnostics snapshot:

- **TRIANGLE remains strongest in this short regime** on family classification and `R` recall, and it also has the lowest mean `R -> S` leakage among the four in this run.
- **Unimodal SimCLR still has much lower matched `R/S` centroid overlap** than the cross-modal methods (better geometry on this pathology metric), so the story is not a single scalar ranking.
- **Pairwise InfoNCE and ConFu are close on several summary metrics** in this short regime, which is exactly the kind of claim that should be reported with standard errors rather than one-run tables.

Short-run caveat:

- The supplementary synergy proxy `R²(y_s12_3)` remains highly unstable and strongly negative in this configuration (large variance across the `n=3` runs), so it should stay out of headline ranking claims in the main text.

### 6.1.3 Lead Readout: Do The Encoders Capture U / R / S?

Before any PID-label confusion matrix is shown, we report a subset-based family classification probe on frozen features. The probe evaluates subsets `x1`, `x2`, `x3`, `x12`, `x13`, `x23`, and `x123`, and asks whether the representation linearly separates the three PID families (unique, redundancy, synergy). We report family accuracy and Cohen's \(\kappa\), with \(\kappa = (p_o-p_e)/(1-p_e)\) and \(\kappa \approx 0\) at chance.

Artifacts (4-model fused comparison):

- `test_outputs/pid_sar3_ssl_fused_confusions/subset_family_probe_heatmaps_four_models.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_four_models_subset_predictors.csv`

We place this readout first because it is a direct representation-level diagnostic: it reveals U/R/S tradeoffs without collapsing them into a single 10-way PID label metric.

Quick snapshot from the current 4-model fused run (`x123` subset only; full subset grid is in the heatmap/CSV above):

| Model | Family-3 acc (`x123`) | Family-3 \(\kappa\) (`x123`) |
| --- | ---: | ---: |
| A: 3x unimodal SimCLR | 0.580 | 0.363 |
| B: pairwise InfoNCE | 0.567 | 0.344 |
| C: TRIANGLE | 0.619 | 0.420 |
| D: ConFu | 0.589 | 0.375 |

This `x123` table is only a compact snapshot. The full subset matrix (`x1`, `x2`, `x3`, `x12`, `x13`, `x23`, `x123`) is the key diagnostic because it shows which methods improve specifically when additional modalities are exposed.

### 6.1.4 Chance-Centered Metric Choice

The primary metric in this results section is **Cohen's kappa**, \(\kappa = (p_o - p_e)/(1 - p_e)\), because it is commonly accepted, chance-corrected, and directly interpretable: \(\kappa \approx 0\) indicates chance-level prediction and \(\kappa = 1\) indicates perfect agreement.

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

This section contains the primary SSL benchmark result. The ranking target is the **source->target modality prediction matrix** in Section `6.8.1`, rather than PID-label classification or latent-proxy `R^2` tables.

The tuned runs used a train/validation/test downstream split with validation-based model selection. In the original tuning run, hyperparameters were selected using validation mean latent-proxy performance (`y_macro_r2`); this is retained for provenance, but the headline evaluation reported here is source->target prediction.

Methods:

1. `A`: 3x unimodal SimCLR
2. `B`: pairwise InfoNCE
3. `C`: TRIANGLE (area contrastive)
4. `D`: ConFu
5. `E`: directional predictive hybrid (`[h_i,h_j] -> h_k`)

Supplementary latent-proxy (`y_*`) regression artifacts are retained for interpretability and tuning provenance, but they are secondary and are discussed in `docs/PID_SAR3_SSL_APPENDIX_ABLATIONS.md`.

### 6.8.1 Source->Target Modality Prediction (Primary Benchmark)

The main downstream benchmark uses modalities directly. Given frozen encoder features for a source subset, we predict a target modality observation. The principal slice is the rotated pair->target setting (`23->1`, `13->2`, `12->3`), but the primary result is the full source->target matrix over sources in `{1,2,3,12,13,23,123}` and targets in `{1,2,3}`.

This design avoids making a hand-crafted latent proxy the primary target. Instead, it asks whether the learned representation supports actual cross-modal prediction.

Main result (one sentence): **under 5-fold macro-\(\kappa\), self/overcomplete source->target tasks remain strong but the cross-modal pair->heldout-target tasks are near chance for all methods, indicating that this regime does not yet support strong chance-corrected cross-modal prediction.**

Presentation order in this section:

- **Figure 14 + Table 7b (the full all-source->target matrix)** are the main benchmark result.
- **Table 7** is a focused excerpt of the three rotated pair->target tasks (`23->1`, `13->2`, `12->3`) and should be read as a slice of `7b`.

Before the full matrix, we report a grouped summary to separate near-ceiling sanity checks from genuinely discriminative cross-modal tasks.

#### Table 7a. Grouped Summary Of The All Source->Target Matrix (macro-\(\kappa\) averages over task groups; A-D only, 5-fold)

Sources:

- `test_outputs/pid_sar3_ssl_fused_confusions/source_to_target_four_models_5fold_grouped_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/source_to_target_four_models_5fold_summary.csv`

| Model | self `1->1/2->2/3->3` | single cross-modal (`1->2`, etc.) | pair->heldout target (`23->1`, `13->2`, `12->3`) | pair->member target (`12->1`, etc.) | `123->target` |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.659 | 0.014 | 0.020 | 0.646 | 0.634 |
| B: pairwise InfoNCE | 0.554 | 0.013 | 0.017 | 0.539 | 0.524 |
| C: TRIANGLE | 0.536 | 0.010 | 0.014 | 0.521 | 0.507 |
| D: ConFu | 0.558 | 0.009 | 0.013 | 0.545 | 0.532 |

For each model \(m\) and task group \(G\), the grouped score is the mean \(\bar{\kappa}_{m,G} = \frac{1}{|G|}\sum_{t\in G} \kappa(m,t)\), where \(t\) indexes source->target tasks in that group.

Main findings from Table 7a:

- Self-prediction and overcomplete settings (`pair->member`, `123->target`) remain strong under \(\kappa\), but they are still sanity checks rather than ranking metrics.
- The discriminative part of the benchmark is the **cross-modal** groups, and under \(\kappa\) these scores are close to zero in the current regime.
- On the grouped pair->heldout-target average, differences are small (`A > B > C > D` here), which argues for improving the experimental regime before making strong method-ranking claims.

Task construction is dimension-wise binary prediction on the target modality. For each target dimension, we threshold the raw target value at the train-split median (which yields approximately balanced classes), fit a linear classifier from frozen source features, and average performance across target dimensions.

For each target dimension, we compute Cohen's \(\kappa_d\) from the binary predictions induced by the train-median threshold. The reported source->target score is the macro average \(\bar{\kappa} = \frac{1}{D}\sum_{d=1}^{D}\kappa_d\), where \(D\) is the target dimensionality.

Evaluation is reported at three levels: (i) the full source->target matrix, (ii) the rotated pair->target slice, and (iii) PID-stratified heatmaps for the rotated slice.

Primary pair->target artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_overall_rotation_scores.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_pid_rotation_scores.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_summary.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_pid_rotation_heatmaps.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1_heatmaps.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/source_to_target_four_models_5fold_summary.csv` (5-fold macro-\(\kappa\); also includes auxiliary scores)
- `test_outputs/pid_sar3_ssl_fused_confusions/source_to_target_four_models_5fold_grouped_summary.csv` (5-fold grouped summary)

Figures 12-14 are legacy visualizations retained for qualitative structure. Tables 7a/7/7b below are the regenerated **5-fold macro-\(\kappa\)** results.

![Rotated pair->target downstream summary](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_summary.png)

*Figure 12. Legacy rotated pair->target summary visualization retained for qualitative comparison (original plot labels use a normalized score).*

![PID-by-rotation pair->target heatmaps](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_pair_to_target_pid_rotation_heatmaps.png)

*Figure 13. Legacy `PID x rotation` pair->target visualization retained for qualitative comparison (original plot labels use a normalized score).*

![All source->target macro-F1 heatmaps](test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1_heatmaps.png)

*Figure 14. Legacy all source->target visualization retained for qualitative structure. The quantitative main result is reported in Tables 7a/7/7b using macro-\(\kappa\).*

#### Table 7. Focused Excerpt From The Main Matrix: Rotated Pair->Target Results (macro-\(\kappa\), 5-fold mean \(\pm\) SE)

Sources:

- `test_outputs/pid_sar3_ssl_fused_confusions/source_to_target_four_models_5fold_summary.csv`

| Model | `23->1` \(\kappa\) | `13->2` \(\kappa\) | `12->3` \(\kappa\) |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.024 ± 0.002 | 0.016 ± 0.005 | 0.021 ± 0.002 |
| B: pairwise InfoNCE | 0.024 ± 0.003 | 0.018 ± 0.004 | 0.009 ± 0.006 |
| C: TRIANGLE | 0.014 ± 0.003 | 0.009 ± 0.005 | 0.019 ± 0.003 |
| D: ConFu | 0.015 ± 0.003 | 0.012 ± 0.002 | 0.012 ± 0.003 |

Rotation-level highlights:

- `23->1`: `A` and `B` are effectively tied at the top within the reported SE (`\kappa \approx 0.024`)
- `13->2`: `B` is highest (`\kappa \approx 0.018`)
- `12->3`: `A` is highest (`\kappa \approx 0.021`), with `C` close (`\kappa \approx 0.019`)

Interpretation of the rotated pair->target slice:

- Under the regenerated 5-fold \(\kappa\) evaluation, all three rotated pair->target tasks are **close to chance** for all methods (\(\kappa\) values near zero).
- The ordering is therefore much less stable and much less important than the stronger conclusion: **the regime needs improvement before chance-corrected cross-modal claims are convincing**.
- The legacy score plots remain useful for qualitative comparison, but they overstate practical separability relative to \(\kappa\).

#### Table 7b. Main Result Matrix: All Source->Target Rotations (macro-\(\kappa\), A-D only, 5-fold means)

Source: `test_outputs/pid_sar3_ssl_fused_confusions/source_to_target_four_models_5fold_summary.csv`

| Source->Target | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| `1->1` | 0.654 | 0.559 | 0.541 | 0.560 |
| `2->1` | 0.017 | 0.014 | 0.001 | 0.005 |
| `3->1` | 0.020 | 0.024 | 0.016 | 0.019 |
| `12->1` | 0.641 | 0.544 | 0.527 | 0.551 |
| `13->1` | 0.643 | 0.543 | 0.525 | 0.544 |
| `23->1` | 0.024 | 0.024 | 0.014 | 0.015 |
| `123->1` | 0.631 | 0.529 | 0.513 | 0.533 |
| `1->2` | 0.011 | 0.010 | 0.004 | 0.014 |
| `2->2` | 0.662 | 0.557 | 0.538 | 0.550 |
| `3->2` | 0.006 | 0.012 | 0.009 | 0.002 |
| `12->2` | 0.646 | 0.543 | 0.519 | 0.536 |
| `13->2` | 0.016 | 0.018 | 0.009 | 0.012 |
| `23->2` | 0.648 | 0.541 | 0.523 | 0.536 |
| `123->2` | 0.634 | 0.526 | 0.505 | 0.522 |
| `1->3` | 0.017 | 0.012 | 0.013 | 0.008 |
| `2->3` | 0.013 | 0.003 | 0.014 | 0.006 |
| `3->3` | 0.662 | 0.547 | 0.530 | 0.564 |
| `12->3` | 0.021 | 0.009 | 0.019 | 0.012 |
| `13->3` | 0.650 | 0.533 | 0.517 | 0.551 |
| `23->3` | 0.650 | 0.531 | 0.518 | 0.550 |
| `123->3` | 0.637 | 0.518 | 0.504 | 0.540 |

Reading guide (how to use the main matrix):

- `1->1`, `2->2`, `3->3` are self-prediction sanity checks.
- `2->1`, `3->1`, `1->2`, ... are single-modality cross-modal transfers.
- `23->1`, `13->2`, `12->3` are the main rotated pair->target tasks summarized in Table 7.

Why `7b` is the main result (rather than only the rotated subset):

- It shows the **full cross-modal behavior surface**, not only three selected tasks.
- It separates ceiling sanity checks (`1->1`, `12->1`, `123->1`, etc.) from the tasks that actually rank methods.
- It makes it harder to overfit the narrative to a small subset of rotations.

Important note on the heuristic “applicable PID” averages:

- We also computed a simple heuristic split of PID atoms into “applicable / non-applicable” for each rotation, but the averages are noisy and not yet a reliable primary metric.
- The full `PID x rotation` heatmaps are more informative than the heuristic scalar summary.
- We also tested a directional predictive hybrid (`E`) in exploratory runs, but it is omitted from the primary table here per the current reporting preference.


### 6.9 What To Do Next (Downstream-First)

1. Add a **synergy-focused tuning track** (select on validation synergy \(\kappa\) instead of all-`y`) and compare with the all-`y` selected models.
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
