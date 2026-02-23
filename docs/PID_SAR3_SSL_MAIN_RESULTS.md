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

### 6.4 Strict Single-Atom Pathology Diagnostics

Before introducing the ladder, we ran targeted diagnostics on the strict single-atom generator to test whether the observed failure was simply due to noise or mixed-atom supervision.

Artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise_redundancy_train_only.csv`

#### Table 6c. Low-Noise (`sigma=0.05`) `pair->heldout` Retrieval, Applicable Split (mean Recall@1 over rotated tasks)

| Model | full-mixture SSL train | redundancy-only SSL train |
| --- | ---: | ---: |
| RAW: observations | 0.0000 | 0.0005 |
| A: 3x unimodal SimCLR | 0.0023 | 0.0009 |
| B: pairwise InfoNCE | 0.0005 | 0.0009 |
| C: TRIANGLE | 0.0009 | 0.0005 |
| D: ConFu | 0.0005 | 0.0005 |

Key diagnosis:

- Low noise does not fix the strict single-atom `pair->heldout` retrieval pathology.
- Restricting SSL training to redundancy atoms does not produce a consistent improvement.
- The raw-observation baseline is itself near random, which shows the issue is not only optimization failure.

### 6.5 Current Strict-Baseline Model Results (Reference Only, Single-Atom Regime)

The results below remain the current reference tables for the strict single-atom style benchmark. They are retained because they are still useful for stress testing and failure-mode analysis, but they should not be treated as the first benchmark to optimize against.

#### 6.5.1 Source->Target Prediction (Cohen's \(\kappa\), 5-fold)

For the source->target benchmark, we predict thresholded raw target coordinates and report macro Cohen's kappa,
\[
\bar{\kappa}=\frac{1}{D}\sum_{d=1}^{D}\kappa_d, \qquad \kappa=\frac{p_o-p_e}{1-p_e}.
\]

##### Table 7a. Grouped Summary Of The Source->Target Matrix (macro-\(\kappa\), 5-fold means)

| Model | self `1->1/2->2/3->3` | single cross-modal | pair->heldout target | pair->member target | `123->target` |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.659 | 0.014 | 0.020 | 0.646 | 0.634 |
| B: pairwise InfoNCE | 0.554 | 0.013 | 0.017 | 0.539 | 0.524 |
| C: TRIANGLE | 0.536 | 0.010 | 0.014 | 0.521 | 0.507 |
| D: ConFu | 0.558 | 0.009 | 0.013 | 0.545 | 0.532 |

##### Table 7. Rotated Pair->Heldout Targets (macro-\(\kappa\), 5-fold mean \(\pm\) SE)

| Model | `23->1` \(\kappa\) | `13->2` \(\kappa\) | `12->3` \(\kappa\) |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.024 ± 0.002 | 0.016 ± 0.005 | 0.021 ± 0.002 |
| B: pairwise InfoNCE | 0.024 ± 0.003 | 0.018 ± 0.004 | 0.009 ± 0.006 |
| C: TRIANGLE | 0.014 ± 0.003 | 0.009 ± 0.005 | 0.019 ± 0.003 |
| D: ConFu | 0.015 ± 0.003 | 0.012 ± 0.002 | 0.012 ± 0.003 |

##### Table 7b. Main Result Matrix: All Source->Target Tasks (macro-\(\kappa\), 5-fold means)

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
    <tr><td><code>1->1</code></td><td align="right" style="background:#d9f2d9;">0.654</td><td align="right" style="background:#d9f2d9;">0.559</td><td align="right" style="background:#d9f2d9;">0.541</td><td align="right" style="background:#d9f2d9;">0.560</td></tr>
    <tr><td><code>1->2</code></td><td align="right" style="background:#f8d7da;">0.011</td><td align="right" style="background:#f8d7da;">0.010</td><td align="right" style="background:#f8d7da;">0.004</td><td align="right" style="background:#f8d7da;">0.014</td></tr>
    <tr><td><code>1->3</code></td><td align="right" style="background:#f8d7da;">0.017</td><td align="right" style="background:#f8d7da;">0.012</td><td align="right" style="background:#f8d7da;">0.013</td><td align="right" style="background:#f8d7da;">0.008</td></tr>
    <tr><td><code>2->1</code></td><td align="right" style="background:#f8d7da;">0.017</td><td align="right" style="background:#f8d7da;">0.014</td><td align="right" style="background:#f8d7da;">0.001</td><td align="right" style="background:#f8d7da;">0.005</td></tr>
    <tr><td><code>2->2</code></td><td align="right" style="background:#d9f2d9;">0.662</td><td align="right" style="background:#d9f2d9;">0.557</td><td align="right" style="background:#d9f2d9;">0.538</td><td align="right" style="background:#d9f2d9;">0.550</td></tr>
    <tr><td><code>2->3</code></td><td align="right" style="background:#f8d7da;">0.013</td><td align="right" style="background:#f8d7da;">0.003</td><td align="right" style="background:#f8d7da;">0.014</td><td align="right" style="background:#f8d7da;">0.006</td></tr>
    <tr><td><code>3->1</code></td><td align="right" style="background:#f8d7da;">0.020</td><td align="right" style="background:#f8d7da;">0.024</td><td align="right" style="background:#f8d7da;">0.016</td><td align="right" style="background:#f8d7da;">0.019</td></tr>
    <tr><td><code>3->2</code></td><td align="right" style="background:#f8d7da;">0.006</td><td align="right" style="background:#f8d7da;">0.012</td><td align="right" style="background:#f8d7da;">0.009</td><td align="right" style="background:#f8d7da;">0.002</td></tr>
    <tr><td><code>3->3</code></td><td align="right" style="background:#d9f2d9;">0.662</td><td align="right" style="background:#d9f2d9;">0.547</td><td align="right" style="background:#d9f2d9;">0.530</td><td align="right" style="background:#d9f2d9;">0.564</td></tr>
    <tr><td><code>12->1</code></td><td align="right" style="background:#d9f2d9;">0.641</td><td align="right" style="background:#d9f2d9;">0.544</td><td align="right" style="background:#d9f2d9;">0.527</td><td align="right" style="background:#d9f2d9;">0.551</td></tr>
    <tr><td><code>12->2</code></td><td align="right" style="background:#d9f2d9;">0.646</td><td align="right" style="background:#d9f2d9;">0.543</td><td align="right" style="background:#d9f2d9;">0.519</td><td align="right" style="background:#d9f2d9;">0.536</td></tr>
    <tr><td><code>12->3</code></td><td align="right" style="background:#f8d7da;">0.021</td><td align="right" style="background:#f8d7da;">0.009</td><td align="right" style="background:#f8d7da;">0.019</td><td align="right" style="background:#f8d7da;">0.012</td></tr>
    <tr><td><code>13->1</code></td><td align="right" style="background:#d9f2d9;">0.643</td><td align="right" style="background:#d9f2d9;">0.543</td><td align="right" style="background:#d9f2d9;">0.525</td><td align="right" style="background:#d9f2d9;">0.544</td></tr>
    <tr><td><code>13->2</code></td><td align="right" style="background:#f8d7da;">0.016</td><td align="right" style="background:#f8d7da;">0.018</td><td align="right" style="background:#f8d7da;">0.009</td><td align="right" style="background:#f8d7da;">0.012</td></tr>
    <tr><td><code>13->3</code></td><td align="right" style="background:#d9f2d9;">0.650</td><td align="right" style="background:#d9f2d9;">0.533</td><td align="right" style="background:#d9f2d9;">0.517</td><td align="right" style="background:#d9f2d9;">0.551</td></tr>
    <tr><td><code>23->1</code></td><td align="right" style="background:#f8d7da;">0.024</td><td align="right" style="background:#f8d7da;">0.024</td><td align="right" style="background:#f8d7da;">0.014</td><td align="right" style="background:#f8d7da;">0.015</td></tr>
    <tr><td><code>23->2</code></td><td align="right" style="background:#d9f2d9;">0.648</td><td align="right" style="background:#d9f2d9;">0.541</td><td align="right" style="background:#d9f2d9;">0.523</td><td align="right" style="background:#d9f2d9;">0.536</td></tr>
    <tr><td><code>23->3</code></td><td align="right" style="background:#d9f2d9;">0.650</td><td align="right" style="background:#d9f2d9;">0.531</td><td align="right" style="background:#d9f2d9;">0.518</td><td align="right" style="background:#d9f2d9;">0.550</td></tr>
    <tr><td><code>123->1</code></td><td align="right" style="background:#d9f2d9;">0.631</td><td align="right" style="background:#d9f2d9;">0.529</td><td align="right" style="background:#d9f2d9;">0.513</td><td align="right" style="background:#d9f2d9;">0.533</td></tr>
    <tr><td><code>123->2</code></td><td align="right" style="background:#d9f2d9;">0.634</td><td align="right" style="background:#d9f2d9;">0.526</td><td align="right" style="background:#d9f2d9;">0.505</td><td align="right" style="background:#d9f2d9;">0.522</td></tr>
    <tr><td><code>123->3</code></td><td align="right" style="background:#d9f2d9;">0.637</td><td align="right" style="background:#d9f2d9;">0.518</td><td align="right" style="background:#d9f2d9;">0.504</td><td align="right" style="background:#d9f2d9;">0.540</td></tr>
  </tbody>
</table>

#### 6.5.2 Strict-Baseline Retrieval And Reconstruction (compact summaries)

Retrieval (single-run frozen-embedding diagnostic, strict baseline):

| Model | pair->heldout `R@1` | pair->member `R@1` | `123->target` `R@1` |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.001 | 0.582 | 0.168 |
| B: pairwise InfoNCE | 0.001 | 0.037 | 0.022 |
| C: TRIANGLE | 0.001 | 0.441 | 0.144 |
| D: ConFu | 0.000 | 0.037 | 0.009 |

Frozen-decoder reconstruction (strict-style regime, 5-fold, macro \(R^2\); pair-source subsets only):

| Decoder | Model | pair->heldout target | pair->member target | `123->target` |
| --- | --- | ---: | ---: | ---: |
| Ridge | A: 3x unimodal SimCLR | -0.233 | 0.681 | 0.645 |
| Ridge | B: pairwise InfoNCE | -0.251 | 0.487 | 0.426 |
| Ridge | C: TRIANGLE | -0.250 | 0.463 | 0.395 |
| Ridge | D: ConFu | -0.238 | 0.530 | 0.477 |
| MLP | A: 3x unimodal SimCLR | -0.141 | 0.415 | 0.341 |
| MLP | B: pairwise InfoNCE | -0.127 | 0.247 | 0.177 |
| MLP | C: TRIANGLE | -0.131 | 0.249 | 0.181 |
| MLP | D: ConFu | -0.120 | 0.274 | 0.204 |

These strict-baseline tables remain useful as a stress test. However, the ladder in Section `6.3` shows they are not the right starting point for objective comparison under exact retrieval.

### 6.6 Rerun Plan (From Benchmarkable To Hard)

The next model reruns should proceed along the compositional ladder first, then be stress-tested on the strict single-atom regime.

1. Rerun the 4-model retrieval benchmark on `L0`, `L1`, and `L2` (same evaluation code, new dataset config).
2. Add a learned frozen-feature `pair->target` retrieval adapter and repeat across `L0 -> L2`.
3. Rerun the source->target \(\kappa\) and reconstruction tables on `L2` first, then back off to `L1` or tighten beyond `L2` as needed.
4. Keep the strict single-atom diagnostics as a separate pathology/stress-test track.

### 6.7 Summary

This revision changes the role of the main results section: it now first establishes a benchmarkable dataset regime and a controlled compositional difficulty ladder, then places the earlier strict single-atom model results in context. The key next step is to rerun the model comparisons on `L0/L1/L2`, with `L2` as the first nontrivial benchmark target.
