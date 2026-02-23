# PID-SAR-3++ SSL Main Results

This document reports the current SSL results after adding a compositional dataset mode and a dataset-side difficulty ladder. The main purpose of this revision is to establish a benchmarkable regime first, then place the earlier strict single-atom results in that context.

## 6. SSL Results (Main Results)

### 6.1 Main Result (Dataset First)

The central result of this revision is a dataset result, not a model ranking result.

1. The original single-atom generator yields a severe `pair->heldout` pathology under exact-instance retrieval: raw observations are near random, even at very low noise.
2. A compositional dataset mode (multi-atom summation + shared backbone) produces a clearly solvable retrieval regime under the same metric.
3. This gives a practical difficulty ladder for rerunning SSL model comparisons in a controlled way, instead of comparing objectives on a pathological benchmark.

The immediate conclusion is that the previous strict benchmark remains scientifically useful as a stress test, but it is not an appropriate starting point for objective comparison under exact retrieval and heldout-modality transfer.

### 6.2 Evaluation Setup (Condensed)

We use frozen-feature downstream evaluations and a same-world split (shared dataset seed for SSL and probes, disjoint train/test samples). For retrieval diagnostics, we report exact-instance retrieval in cosine space using Recall@\(K\),
\[
\mathrm{R@}K = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\mathrm{rank}_i \le K],
\]
and mean reciprocal rank,
\[
\mathrm{MRR} = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{\mathrm{rank}_i}.
\]

Unless otherwise noted, `pair->heldout` refers to the rotated tasks `23->1`, `13->2`, `12->3`.

### 6.3 Dataset Difficulty Ladder (Primary Result For This Revision)

We added a backward-compatible compositional dataset mode with the following new controls:

- `composition_mode` (`single_atom` / `multi_atom`)
- `active_atoms_per_sample`
- `shared_backbone_gain`
- `shared_backbone_tied_projection`
- `synergy_deleak_lambda`

The ladder below evaluates **RAW exact-instance retrieval** (no learned encoder) to isolate benchmarkability before rerunning full SSL comparisons.

Artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/dataset_difficulty_ladder_raw_retrieval.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/dataset_difficulty_ladder_raw_retrieval_grouped.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/dataset_difficulty_ladder_raw_retrieval.png`

Random baseline for `Recall@1` with `n=1800` gallery items is `1/1800 ≈ 0.00056`.

#### Table 6a. Ladder Definition (dataset-side knobs)

| Level | Setting | `sigma` | `active_atoms` | `shared_gain` | shared proj tied? | `synergy_deleak_lambda` |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| L0 | `compositional_very_easy` | 0.02 | 5 | 4.0 | yes | 0.25 |
| L1 | `compositional_easy_plus` | 0.025 | 4 | 3.2 | yes | 0.35 |
| L2 | `compositional_easy` | 0.03 | 4 | 2.5 | yes | 0.5 |

#### Table 6b. RAW Retrieval Across The Difficulty Ladder (group means, Recall@1)

| Level | pair->heldout | pair->member | `123->target` |
| --- | ---: | ---: | ---: |
| L0 `compositional_very_easy` | 0.6487 | 0.9976 | 0.9624 |
| L1 `compositional_easy_plus` | 0.5839 | 0.9970 | 0.9511 |
| L2 `compositional_easy` | 0.3856 | 0.9924 | 0.9002 |

Main interpretation of the ladder:

- `L0 -> L1 -> L2` provides a clean monotonic progression on `pair->heldout` exact retrieval (`0.649 -> 0.584 -> 0.386` in raw `Recall@1`).
- All three levels are decisively above random (`≈ 0.00056`) and therefore benchmarkable under the current retrieval metric.
- `L2` remains challenging enough to be useful as a first nontrivial benchmark level, while `L0/L1` are calibration levels for debugging objectives and probes.

This ladder is the correct starting point for the next model reruns: it lets us compare objectives while controlling dataset difficulty explicitly.

### 6.4 Compositional-Very-Easy Model Rerun (L0)

After establishing the ladder, we reran the main downstream analyses on `L0 = compositional_very_easy` to verify that the objectives separate in a benchmarkable regime.

This rerun is a compact pass (CPU-friendly) intended to update the qualitative conclusions quickly:

- SSL training: `120` steps
- probe set: `40` samples per primary PID label (`n=400`)
- kappa and reconstruction: `2` folds
- kappa evaluation restricted to source subsets `{12,13,23,123}` (the post-6.3 task focus)

Artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/compositional_very_easy_source_to_target_four_models_5fold_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/compositional_very_easy_source_to_target_four_models_5fold_grouped_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/compositional_very_easy_retrieval_source_to_target_four_models_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/compositional_very_easy_source_to_target_reconstruction_four_models_5fold_summary.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/compositional_very_easy_source_to_target_reconstruction_four_models_5fold_grouped_summary.csv`

#### 6.4.1 Source->Target Prediction (Cohen's \(\kappa\), compositional `L0` quick rerun)

For the source->target benchmark, we predict thresholded raw target coordinates and report macro Cohen's kappa,
\[
\bar{\kappa}=\frac{1}{D}\sum_{d=1}^{D}\kappa_d, \qquad \kappa=\frac{p_o-p_e}{1-p_e}.
\]

##### Table 7a. Grouped Summary (compositional `L0`, macro-\(\kappa\))

| Model | pair->heldout target | pair->member target | `123->target` |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.631 | 0.709 | 0.701 |
| B: pairwise InfoNCE | 0.639 | 0.682 | 0.674 |
| C: TRIANGLE | 0.624 | 0.676 | 0.669 |
| D: ConFu | 0.623 | 0.669 | 0.663 |

In `L0`, the benchmark is solvable and chance-corrected performance is high across all three groups, which is the intended behavior of a calibration regime.

##### Table 7. Rotated Pair->Heldout Targets (compositional `L0`, macro-\(\kappa\), 2-fold mean \(\pm\) SE)

| Model | `23->1` \(\kappa\) | `13->2` \(\kappa\) | `12->3` \(\kappa\) |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.623 ± 0.001 | 0.634 ± 0.001 | 0.636 ± 0.011 |
| B: pairwise InfoNCE | 0.637 ± 0.008 | 0.642 ± 0.002 | 0.639 ± 0.011 |
| C: TRIANGLE | 0.620 ± 0.004 | 0.629 ± 0.001 | 0.624 ± 0.009 |
| D: ConFu | 0.615 ± 0.002 | 0.627 ± 0.004 | 0.628 ± 0.010 |

##### Table 7b. Pair/Triple Source->Target Matrix (compositional `L0`, macro-\(\kappa\), 2-fold means)

Cell colors use a fixed threshold at \(\kappa=0.25\): green for \(\kappa>0.25\), red for \(\kappa\le 0.25\).

<table>
  <thead>
    <tr>
      <th align="left">Source-&gt;Target</th>
      <th align="right">A</th>
      <th align="right">B</th>
      <th align="right">C</th>
      <th align="right">D</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>12->1</code></td><td align="right" style="background:#d9f2d9;">0.710</td><td align="right" style="background:#d9f2d9;">0.687</td><td align="right" style="background:#d9f2d9;">0.693</td><td align="right" style="background:#d9f2d9;">0.663</td></tr>
    <tr><td><code>12->2</code></td><td align="right" style="background:#d9f2d9;">0.708</td><td align="right" style="background:#d9f2d9;">0.680</td><td align="right" style="background:#d9f2d9;">0.668</td><td align="right" style="background:#d9f2d9;">0.674</td></tr>
    <tr><td><code>12->3</code></td><td align="right" style="background:#d9f2d9;">0.636</td><td align="right" style="background:#d9f2d9;">0.639</td><td align="right" style="background:#d9f2d9;">0.624</td><td align="right" style="background:#d9f2d9;">0.628</td></tr>
    <tr><td><code>13->1</code></td><td align="right" style="background:#d9f2d9;">0.706</td><td align="right" style="background:#d9f2d9;">0.683</td><td align="right" style="background:#d9f2d9;">0.683</td><td align="right" style="background:#d9f2d9;">0.669</td></tr>
    <tr><td><code>13->2</code></td><td align="right" style="background:#d9f2d9;">0.634</td><td align="right" style="background:#d9f2d9;">0.642</td><td align="right" style="background:#d9f2d9;">0.629</td><td align="right" style="background:#d9f2d9;">0.627</td></tr>
    <tr><td><code>13->3</code></td><td align="right" style="background:#d9f2d9;">0.710</td><td align="right" style="background:#d9f2d9;">0.684</td><td align="right" style="background:#d9f2d9;">0.670</td><td align="right" style="background:#d9f2d9;">0.667</td></tr>
    <tr><td><code>23->1</code></td><td align="right" style="background:#d9f2d9;">0.623</td><td align="right" style="background:#d9f2d9;">0.637</td><td align="right" style="background:#d9f2d9;">0.620</td><td align="right" style="background:#d9f2d9;">0.615</td></tr>
    <tr><td><code>23->2</code></td><td align="right" style="background:#d9f2d9;">0.712</td><td align="right" style="background:#d9f2d9;">0.681</td><td align="right" style="background:#d9f2d9;">0.671</td><td align="right" style="background:#d9f2d9;">0.679</td></tr>
    <tr><td><code>23->3</code></td><td align="right" style="background:#d9f2d9;">0.710</td><td align="right" style="background:#d9f2d9;">0.678</td><td align="right" style="background:#d9f2d9;">0.670</td><td align="right" style="background:#d9f2d9;">0.665</td></tr>
    <tr><td><code>123->1</code></td><td align="right" style="background:#d9f2d9;">0.697</td><td align="right" style="background:#d9f2d9;">0.674</td><td align="right" style="background:#d9f2d9;">0.677</td><td align="right" style="background:#d9f2d9;">0.662</td></tr>
    <tr><td><code>123->2</code></td><td align="right" style="background:#d9f2d9;">0.701</td><td align="right" style="background:#d9f2d9;">0.675</td><td align="right" style="background:#d9f2d9;">0.663</td><td align="right" style="background:#d9f2d9;">0.665</td></tr>
    <tr><td><code>123->3</code></td><td align="right" style="background:#d9f2d9;">0.706</td><td align="right" style="background:#d9f2d9;">0.673</td><td align="right" style="background:#d9f2d9;">0.667</td><td align="right" style="background:#d9f2d9;">0.664</td></tr>
  </tbody>
</table>

Main point for `L0`: the downstream benchmark is no longer collapsed, and the methods can be meaningfully separated.

#### 6.4.2 Frozen Retrieval (compositional `L0`, single run)

Retrieval in `L0` is a much stronger diagnostic than in the strict single-atom regime because the task is benchmarkable under the current metric.

| Model | pair->heldout `R@1` | pair->member `R@1` | `123->target` `R@1` |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.2700 | 0.8575 | 0.7025 |
| B: pairwise InfoNCE | 0.0025 | 0.5629 | 0.3092 |
| C: TRIANGLE | 0.0017 | 0.2450 | 0.1483 |
| D: ConFu | 0.0025 | 0.5513 | 0.2892 |

This reveals a strong separation in the current `L0` setting: unimodal SimCLR dominates exact retrieval, while the contrastive fusion variants lag substantially on this metric.

#### 6.4.3 Frozen-Decoder Reconstruction (compositional `L0`, 2-fold)

We retain the same reconstruction benchmark definition and report macro \(R^2\), where positive values indicate better-than-baseline reconstruction and higher is better.

Grouped summary (macro \(R^2\)):

| Decoder | Model | pair->heldout target | pair->member target | `123->target` |
| --- | --- | ---: | ---: | ---: |
| Ridge | A: 3x unimodal SimCLR | 0.698 | 0.833 | 0.817 |
| Ridge | B: pairwise InfoNCE | 0.697 | 0.798 | 0.777 |
| Ridge | C: TRIANGLE | 0.692 | 0.793 | 0.782 |
| Ridge | D: ConFu | 0.692 | 0.791 | 0.777 |
| MLP | A: 3x unimodal SimCLR | 0.632 | 0.690 | 0.677 |
| MLP | B: pairwise InfoNCE | 0.632 | 0.662 | 0.653 |
| MLP | C: TRIANGLE | 0.619 | 0.654 | 0.638 |
| MLP | D: ConFu | 0.612 | 0.641 | 0.634 |

Rotated `pair->heldout` slice (macro \(R^2\), 2-fold mean \(\pm\) SE):

| Decoder | Model | `23->1` | `13->2` | `12->3` |
| --- | --- | ---: | ---: | ---: |
| Ridge | A: 3x unimodal SimCLR | 0.700 ± 0.008 | 0.696 ± 0.004 | 0.698 ± 0.000 |
| Ridge | B: pairwise InfoNCE | 0.690 ± 0.001 | 0.701 ± 0.010 | 0.701 ± 0.003 |
| Ridge | C: TRIANGLE | 0.687 ± 0.004 | 0.700 ± 0.009 | 0.689 ± 0.001 |
| Ridge | D: ConFu | 0.689 ± 0.000 | 0.693 ± 0.001 | 0.695 ± 0.001 |
| MLP | A: 3x unimodal SimCLR | 0.628 ± 0.008 | 0.632 ± 0.002 | 0.635 ± 0.002 |
| MLP | B: pairwise InfoNCE | 0.628 ± 0.002 | 0.632 ± 0.008 | 0.636 ± 0.000 |
| MLP | C: TRIANGLE | 0.623 ± 0.004 | 0.618 ± 0.009 | 0.617 ± 0.003 |
| MLP | D: ConFu | 0.609 ± 0.001 | 0.614 ± 0.014 | 0.612 ± 0.001 |

In `L0`, reconstruction is uniformly strong and no longer dominated by near-chance behavior. This confirms that the new compositional regime is suitable for objective comparison.

### 6.5 Strict Single-Atom Pathology Diagnostics (Reference)

We retain the strict single-atom pathology diagnostics as a separate stress-test track.

Artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise_redundancy_train_only.csv`

Low-noise (`sigma=0.05`) applicable-split `pair->heldout` retrieval remains near-random even for raw observations, confirming that the old single-atom regime should be treated as a pathology probe rather than the first objective-comparison benchmark.

### 6.6 Rerun Plan (From Very Easy To Nontrivial)

Now that `L0` is benchmarkable and the full post-6.3 analysis has been rerun there, the next step is to move the same analysis stack to `L1` and `L2`.

1. Rerun the compositional analysis bundle on `L1` and `L2` (same outputs: kappa/retrieval/reconstruction).
2. Track where the current objective ranking starts to change between `L0`, `L1`, and `L2`.
3. Add a learned frozen-feature `pair->target` retrieval adapter and repeat on `L0 -> L2`.
4. Keep strict single-atom diagnostics as a separate pathology/stress-test track.

### 6.7 Summary

The post-6.3 analyses now run on a benchmarkable compositional regime (`L0`) rather than the pathological single-atom regime. This changes the interpretation of the results: objective differences can now be measured in a solvable setting, while the strict single-atom generator is retained as a deliberate stress test.
