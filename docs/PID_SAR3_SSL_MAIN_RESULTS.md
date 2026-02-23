# PID-SAR-3++ SSL Main Results

This document reports the main SSL results for PID-SAR-3++ using a same-world evaluation split, frozen encoders, and chance-corrected downstream metrics. Supplementary ablations and legacy analyses are moved to `docs/PID_SAR3_SSL_APPENDIX_ABLATIONS.md`.

## 6. SSL Results (Main Results)

### 6.1 Main Result (What Matters)

Across three complementary validations, the same conclusion appears.

1. In the primary source->target benchmark, the rotated `pair->heldout` tasks (`23->1`, `13->2`, `12->3`) are near chance under Cohen's kappa (\(\kappa\approx 0\)) for all methods.
2. Retrieval (which is closer to the contrastive training objective) confirms that the encoders preserve strong instance alignment for self and overcomplete tasks, but not for `pair->heldout` transfer.
3. Frozen-decoder reconstruction with both linear and nonlinear decoders improves easy settings, but `pair->heldout` reconstruction remains below baseline (negative macro \(R^2\)).

The main scientific conclusion is therefore not a model ranking claim. It is that the current regime does not yet produce robust cross-modal transfer to a heldout modality in a chance-corrected or reconstruction-based evaluation.

### 6.2 Evaluation Protocol (Condensed)

All results here use a same-world split: SSL training and downstream probing share the same dataset seed, while probe train/test examples are disjoint samples from that same world. This avoids cross-seed dictionary changes from contaminating the evaluation.

Encoders are frozen. For source subsets, we concatenate the corresponding frozen modality embeddings. Unless otherwise noted, uncertainty is reported as mean \(\pm\) standard error, with \(\mathrm{SE}=s/\sqrt{n}\).

### 6.3 Primary Benchmark: Source->Target Prediction (Cohen's \(\kappa\))

For each source->target task, we predict the raw target modality dimension-wise after train-median binarization and report the macro average of per-dimension Cohen's kappa,
\[
\bar{\kappa}=\frac{1}{D}\sum_{d=1}^{D}\kappa_d,\qquad \kappa=\frac{p_o-p_e}{1-p_e},
\]
where \(D\) is the target dimensionality, \(p_o\) is observed agreement, and \(p_e\) is chance agreement.

#### Table 7a. Grouped Summary Of The Source->Target Matrix (macro-\(\kappa\), 5-fold means)

| Model | self `1->1/2->2/3->3` | single cross-modal (`1->2`, etc.) | pair->heldout target (`23->1`, `13->2`, `12->3`) | pair->member target (`12->1`, etc.) | `123->target` |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.659 | 0.014 | 0.020 | 0.646 | 0.634 |
| B: pairwise InfoNCE | 0.554 | 0.013 | 0.017 | 0.539 | 0.524 |
| C: TRIANGLE | 0.536 | 0.010 | 0.014 | 0.521 | 0.507 |
| D: ConFu | 0.558 | 0.009 | 0.013 | 0.545 | 0.532 |

The grouped average for model \(m\) and task group \(G\) is \(\bar{\kappa}_{m,G}=|G|^{-1}\sum_{t\in G}\kappa(m,t)\).

#### Table 7. Rotated Pair->Heldout Targets (macro-\(\kappa\), 5-fold mean \(\pm\) SE)

| Model | `23->1` \(\kappa\) | `13->2` \(\kappa\) | `12->3` \(\kappa\) |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.024 ± 0.002 | 0.016 ± 0.005 | 0.021 ± 0.002 |
| B: pairwise InfoNCE | 0.024 ± 0.003 | 0.018 ± 0.004 | 0.009 ± 0.006 |
| C: TRIANGLE | 0.014 ± 0.003 | 0.009 ± 0.005 | 0.019 ± 0.003 |
| D: ConFu | 0.015 ± 0.003 | 0.012 ± 0.002 | 0.012 ± 0.003 |

Interpretation: all four methods are near chance on the discriminative cross-modal tasks under a chance-corrected metric.

#### Table 7b. Main Result Matrix: All Source->Target Tasks (macro-\(\kappa\), 5-fold means)

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

### 6.4 Retrieval Validation (Frozen Embeddings, Instance Retrieval)

To test what the contrastive encoders preserve without forcing a coordinate decoder, we evaluate source->target retrieval in frozen embedding space. For a query embedding, the positive is the matched sample in the target gallery. We report Recall@\(K\),
\(\mathrm{R@}K = N^{-1}\sum_i \mathbf{1}[\mathrm{rank}_i \le K]\), and mean reciprocal rank, \(\mathrm{MRR}=N^{-1}\sum_i \mathrm{rank}_i^{-1}\).

This retrieval benchmark is currently a single-run diagnostic (same-world setting), so we use it for structure and failure analysis rather than final ranking.

#### Retrieval Summary (group means, Recall@1)

| Model | self | single cross-modal | pair->heldout | pair->member | `123->target` |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 1.000 | 0.001 | 0.001 | 0.582 | 0.168 |
| B: pairwise InfoNCE | 1.000 | 0.000 | 0.001 | 0.037 | 0.022 |
| C: TRIANGLE | 1.000 | 0.000 | 0.001 | 0.441 | 0.144 |
| D: ConFu | 1.000 | 0.000 | 0.000 | 0.037 | 0.009 |

#### Retrieval Focused Slice (rotated `pair->heldout`, Recall@1)

| Model | `23->1` | `13->2` | `12->3` |
| --- | ---: | ---: | ---: |
| A: 3x unimodal SimCLR | 0.002 | 0.000 | 0.000 |
| B: pairwise InfoNCE | 0.001 | 0.001 | 0.001 |
| C: TRIANGLE | 0.001 | 0.001 | 0.001 |
| D: ConFu | 0.000 | 0.000 | 0.001 |

Retrieval therefore supports the same core conclusion as Table 7: the hard cross-modal transfer slice is not solved, even when evaluated in an objective-aligned way.

### 6.5 Frozen-Decoder Reconstruction (Raw Target Modalities)

We next test whether richer decoders can recover the heldout modality from frozen encoder features. We train multi-output decoders on frozen source features and reconstruct the raw target modality vector. We report macro \(R^2\),
\[
\bar{R}^2 = \frac{1}{D}\sum_{d=1}^{D} R_d^2,
\]
where \(R_d^2=1-\sum_i(y_{id}-\hat y_{id})^2 / \sum_i(y_{id}-\bar y_d)^2\). The baseline predictor (train-mean target) gives \(R^2\approx 0\); negative values indicate worse-than-baseline reconstruction.

Decoders:

- `Ridge`: linear multi-output regression
- `MLP`: nonlinear multi-output regression (frozen encoders, trainable decoder only)

#### Reconstruction Summary (group means, macro \(R^2\))

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

#### Reconstruction Focused Slice (rotated `pair->heldout`, macro \(R^2\), 5-fold mean \(\pm\) SE)

| Decoder | Model | `23->1` | `13->2` | `12->3` |
| --- | --- | ---: | ---: | ---: |
| Ridge | A: 3x unimodal SimCLR | -0.232 ± 0.018 | -0.237 ± 0.011 | -0.231 ± 0.008 |
| Ridge | B: pairwise InfoNCE | -0.248 ± 0.007 | -0.246 ± 0.012 | -0.259 ± 0.013 |
| Ridge | C: TRIANGLE | -0.238 ± 0.011 | -0.264 ± 0.011 | -0.247 ± 0.008 |
| Ridge | D: ConFu | -0.223 ± 0.009 | -0.246 ± 0.008 | -0.244 ± 0.009 |
| MLP | A: 3x unimodal SimCLR | -0.140 ± 0.004 | -0.142 ± 0.005 | -0.139 ± 0.008 |
| MLP | B: pairwise InfoNCE | -0.126 ± 0.004 | -0.129 ± 0.008 | -0.125 ± 0.009 |
| MLP | C: TRIANGLE | -0.134 ± 0.005 | -0.132 ± 0.010 | -0.126 ± 0.004 |
| MLP | D: ConFu | -0.110 ± 0.005 | -0.125 ± 0.006 | -0.125 ± 0.003 |

Main reconstruction takeaway: a nonlinear decoder improves the hard `pair->heldout` slice relative to linear decoding (less negative \(R^2\)), but the frozen encoders still do not support baseline-beating heldout-modality reconstruction in this regime.

### 6.6 Compact Representation Sanity Check (U/R/S Family Probe)

As a representation-level sanity check, we evaluate a frozen linear probe on the `x123` subset for 3-way PID-family prediction (Unique / Redundancy / Synergy), reporting accuracy and Cohen's \(\kappa\).

| Model | Family-3 acc (`x123`) | Family-3 \(\kappa\) (`x123`) |
| --- | ---: | ---: |
| A: 3x unimodal SimCLR | 0.580 | 0.363 |
| B: pairwise InfoNCE | 0.568 | 0.344 |
| C: TRIANGLE | 0.619 | 0.420 |
| D: ConFu | 0.589 | 0.375 |

This sanity check shows that the encoders do capture PID-family structure, but Tables 7, retrieval, and reconstruction show that this does not yet translate into strong heldout-modality transfer.

### 6.7 Pathology Diagnosis (Low-Noise + Redundancy-Only Training)

To test whether the `pair->heldout` failure is merely a noise problem, we ran a low-noise diagnostic with `sigma=0.05` and split the rotated tasks into `applicable` vs `non_applicable` PID atoms under the single-atom generator. We also repeated the diagnostic with SSL training restricted to redundancy atoms only (`R12/R13/R23/R123`).

Diagnostic artifacts:

- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise.csv`
- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise.png`
- `test_outputs/pid_sar3_ssl_fused_confusions/pair_to_heldout_retrieval_applicability_low_noise_redundancy_train_only.csv`

The exact-instance retrieval baseline is extremely strict here: with `n=1800` gallery items, random `Recall@1` is approximately `1/1800 = 0.00056`.

#### Low-noise pathology summary (mean over rotated `pair->heldout` tasks, Recall@1)

| Model | Full-mixture train, applicable | Redundancy-only train, applicable |
| --- | ---: | ---: |
| RAW: observations | 0.0000 | 0.0005 |
| A: 3x unimodal SimCLR | 0.0023 | 0.0009 |
| B: pairwise InfoNCE | 0.0005 | 0.0009 |
| C: TRIANGLE | 0.0009 | 0.0005 |
| D: ConFu | 0.0005 | 0.0005 |

Main diagnosis:

- Lowering observation noise does **not** rescue exact `pair->heldout` retrieval.
- Restricting SSL training to redundancy atoms does **not** produce a clear improvement on applicable `pair->heldout` retrieval.
- The `RAW` baseline is also near-random, which indicates the current exact-retrieval formulation is itself a poor proxy for recoverability in the single-atom generator (not only a learned-representation failure).

This supports the interpretation that the present pathology is a combination of dataset structure (single-atom samples, many inapplicable cases) and evaluation mismatch (exact instance retrieval with simple pair fusion), not just optimization failure.

### 6.8 Summary

The results section is intentionally centered on one question: do frozen encoders support robust cross-modal transfer to a heldout modality? Under three different validations (chance-corrected prediction, retrieval, and reconstruction), the answer is currently no. The next iteration should target the `pair->heldout` slice directly in model selection and training.
