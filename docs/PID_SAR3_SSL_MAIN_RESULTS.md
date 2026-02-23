# PID-SAR-3++ SSL Main Results

This document reports the primary SSL results for PID-SAR-3++. It presents the corrected evaluation protocol, a compact set of representation-level diagnostics, and the source->target downstream benchmark used for method ranking. Hyperparameter sweeps, supplementary latent-proxy analyses, and broader ablations are reported separately in `docs/PID_SAR3_SSL_APPENDIX_ABLATIONS.md`.

## 6. SSL Results (Main Results)

The section numbering below is preserved from the original SSL report to keep table/figure references stable.

### 6.1 Evaluation Protocol (Important Correction)

Earlier comparisons used different dataset seeds for probe-train and probe-test generators. In PID-SAR-3++, changing the dataset seed changes the fixed projection operators, the fixed synergy network, and the de-leakage maps, so cross-seed probing evaluates transfer across different observation dictionaries rather than ordinary generalization to new samples from the same world.

All results reported here therefore use a same-world split: the SSL model and probe splits share the same dataset seed, while train/test probe examples are sampled independently. Encoders are frozen at evaluation time, the three modality embeddings are concatenated as `[h_1,h_2,h_3]`, and linear probes are fit on held-out data.

Implementation:

- `tests/test_pid_sar3_ssl_fused_confusions.py`

### 6.1.1 Main-Results Reporting Contract

To avoid presenting the benchmark as a single-seed leaderboard, the main results are defined as a small set of decision metrics reported with uncertainty across repeated runs. In practice, we report the sample mean \(\bar{x}\) together with the standard error \(\mathrm{SE}=s/\sqrt{n}\), where \(s\) is the sample standard deviation over runs and \(n\) is the number of runs.

The main text prioritizes one ranking target (the source->target matrix in Section `6.8.1`) together with a small number of failure-mode diagnostics, while broader sweeps and auxiliary tables are moved to the appendix.

Implementation artifact (repeated-seed summary):

- `test_outputs/pid_sar3_ssl_fused_confusions/main_results_four_models_seeded_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/main_results_four_models_seeded_trials.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/main_results_four_models_seeded_summary.png`

### 6.1.2 Repeated-Seed Secondary Diagnostics Snapshot (Quick CPU Run, `n=3`)

We ran the repeated-seed summary harness (`test_main_results_four_models_repeated_seed_summary`) on February 23, 2026 in a short CPU regime (`n=3` dataset worlds / optimization seeds; `140` SSL steps). This run is not the final benchmark, but it is sufficient to replace single-seed statements with uncertainty-aware summaries.

Secondary diagnostics (mean \(\pm\) SE):

| Model | Family-3 acc | mean `R` recall | mean `R -> S` leakage | mean matched `R/S` centroid cos |
| --- | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.566 ± 0.0068 | 0.546 ± 0.0135 | 0.201 ± 0.0094 | 0.769 ± 0.0227 |
| B: pairwise InfoNCE | 0.560 ± 0.0010 | 0.520 ± 0.0059 | 0.200 ± 0.0108 | 0.921 ± 0.0095 |
| C: TRIANGLE | 0.606 ± 0.0129 | 0.642 ± 0.0076 | 0.186 ± 0.0162 | 0.936 ± 0.0122 |
| D: ConFu | 0.557 ± 0.0028 | 0.508 ± 0.0161 | 0.202 ± 0.0072 | 0.928 ± 0.0101 |

Interpretation of the secondary diagnostics snapshot:

- **TRIANGLE remains strongest in this short regime** on family classification and `R` recall, and it also has the lowest mean `R -> S` leakage among the four in this run.
- **Unimodal SimCLR still has much lower matched `R/S` centroid overlap** than the cross-modal methods (better geometry on this pathology metric), so the story is not a single scalar ranking.
- **Pairwise InfoNCE and ConFu are close on several summary metrics** in this short regime, which is exactly the kind of claim that should be reported with standard errors rather than one-run tables.

Short-run caveat:

- The supplementary synergy proxy `R²(y_s12_3)` remains highly unstable and strongly negative in this configuration (large variance across the `n=3` runs), so it should stay out of headline ranking claims in the main text.

### 6.1.3 Lead Readout: Do The Encoders Capture U / R / S?

Before any PID-label confusion matrix is shown, we report a subset-based family classification probe on frozen features. The probe evaluates subsets `x1`, `x2`, `x3`, `x12`, `x13`, `x23`, and `x123`, and asks whether the representation linearly separates the three PID families (unique, redundancy, synergy).

For clarity, if precision and recall are denoted by \(P\) and \(R\), the class-wise F1 score is \(F_1 = 2PR/(P+R)\). The reported family macro-F1 is the unweighted mean \(\mathrm{macro\text{-}F1} = \frac{1}{3}\sum_{c\in\{U,R,S\}}F_1^{(c)}\), and the `U`/`R`/`S` columns report one-vs-rest scores \(F_1^{(c)}\) for each family.

Artifacts (4-model fused comparison):

- `test_outputs/pid_sar3_ssl_fused_confusions/subset_family_probe_heatmaps_four_models.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/fused_frozen_four_models_subset_predictors.csv`

We place this readout first because it is a direct representation-level diagnostic: it reveals U/R/S tradeoffs without collapsing them into a single 10-way PID label metric.

Quick snapshot from the current 4-model fused run (`x123` subset only; full subset grid is in the heatmap/CSV above):

| Model | Family-3 acc (`x123`) | Family-3 macro-F1 (`x123`) | `U` F1 (`x123`) | `R` F1 (`x123`) | `S` F1 (`x123`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.586 | 0.580 | 0.609 | 0.620 | 0.510 |
| B: pairwise InfoNCE | 0.567 | 0.560 | 0.608 | 0.602 | 0.469 |
| C: TRIANGLE | 0.619 | 0.611 | 0.636 | 0.664 | 0.533 |
| D: ConFu | 0.589 | 0.582 | 0.583 | 0.634 | 0.530 |

This `x123` table is only a compact snapshot. The full subset matrix (`x1`, `x2`, `x3`, `x12`, `x13`, `x23`, `x123`) is the key diagnostic because it shows which methods improve specifically when additional modalities are exposed.

### 6.1.4 Chance-Centered Alternatives To F1 (Recommendation)

If we want a metric whose random baseline is exactly \(0\) without applying a post-hoc skill transform, the most practical replacement for F1 in this section is **Cohen's kappa**, defined as \(\kappa = (p_o - p_e)/(1 - p_e)\), where \(p_o\) is observed agreement and \(p_e\) is chance agreement under the empirical marginals. By construction, chance-level performance gives \(\kappa \approx 0\).

Two additional options are also reasonable:

- **Matthews correlation coefficient (MCC)** for binary tasks (and its multiclass extension), which is also centered near \(0\) at chance.
- **Chance-corrected F1 (F1-skill)**, e.g. \(F_{1,\mathrm{skill}} = \frac{F_1 - F_{1,\mathrm{rand}}}{1 - F_{1,\mathrm{rand}}}\); for balanced binary tasks with \(F_{1,\mathrm{rand}} \approx 0.5\), this reduces to \(F_{1,\mathrm{skill}} \approx 2F_1 - 1\).

For the source->target benchmark in Section `6.8.1`, a good reporting practice is to keep macro-F1 for comparability with prior plots and add \(\kappa\) as the chance-centered primary scalar.

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

Presentation order in this section:

- **Figure 14 + Table 7b (the full all-source->target matrix)** are the main benchmark result.
- **Table 7** is a focused excerpt of the three rotated pair->target tasks (`23->1`, `13->2`, `12->3`) and should be read as a slice of `7b`.

Before the full matrix, we report a grouped summary to separate near-ceiling sanity checks from genuinely discriminative cross-modal tasks.

#### Table 7a. Grouped Summary Of The All Source->Target Matrix (macro-F1 averages over task groups; A-D only)

Source: `test_outputs/pid_sar3_ssl_fused_confusions/tuned_long_steps_600_all_source_to_target_macro_f1.csv`

| Model | self `1->1/2->2/3->3` | single cross-modal (`1->2`, etc.) | pair->heldout target (`23->1`, `13->2`, `12->3`) | pair->member target (`12->1`, etc.) | `123->target` |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.987 | 0.511 | 0.516 | 0.984 | 0.981 |
| B: pairwise InfoNCE | 0.986 | 0.614 | 0.639 | 0.982 | 0.978 |
| C: TRIANGLE | 0.986 | 0.615 | 0.651 | 0.982 | 0.979 |
| D: ConFu | 0.986 | 0.600 | 0.629 | 0.982 | 0.979 |

For each model \(m\) and task group \(G\), the grouped score is the mean \(\bar{F}_{m,G} = \frac{1}{|G|}\sum_{t\in G} F_1(m,t)\), where \(t\) indexes source->target tasks in that group.

Main findings from Table 7a:

- Self-prediction and overcomplete settings (`pair->member`, `123->target`) are near-ceiling for all methods, so they are sanity checks, not ranking metrics.
- The ranking signal lives in the **cross-modal** groups, especially **pair->heldout target**.
- `C: TRIANGLE` is strongest on the grouped pair->heldout target average; `B: pairwise InfoNCE` is close; `D: ConFu` remains competitive; `A` is near the cross-modal floor.

Task construction is dimension-wise binary prediction on the target modality. For each target dimension, we threshold the raw target value at the train-split median (which yields approximately balanced classes), fit a linear classifier from frozen source features, and average performance across target dimensions.

Let \(F_{1,d}\) denote the binary F1 score for target dimension \(d\). The reported macro-F1 is \(\frac{1}{D}\sum_{d=1}^{D}F_{1,d}\). We also report Cohen's \(\kappa\), \(\kappa = (p_o-p_e)/(1-p_e)\), as a chance-centered metric (\(\kappa \approx 0\) at random performance), and the normalized score \(F_{1,\mathrm{skill}} = (F_1-0.5)/0.5 = 2F_1-1\) for the median-balanced binary tasks.

Evaluation is reported at three levels: (i) the full source->target matrix, (ii) the rotated pair->target slice, and (iii) PID-stratified heatmaps for the rotated slice.

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

*Figure 14. Main downstream result: all source->target rotations (`1/2/3/12/13/23/123 -> 1/2/3`) reported as macro-F1 for A-D (frozen encoders). Includes self-prediction rows such as `1->1` and cross-modal rows such as `2->1`.*

#### Table 7. Focused Excerpt From The Main Matrix: Rotated Pair->Target Results (frozen encoders, held-out test)

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

Interpretation of the rotated pair->target slice:

- **Cross-modal methods clearly outperform unimodal SimCLR** on the pair->target tasks (model A remains near the cross-modal floor).
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

Why `7b` is the main result (rather than only the rotated subset):

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
