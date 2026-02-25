# Claude Session Notes

## Project Summary

Synthetic 3-view PID (Partial Information Decomposition) SSL benchmark.
Goal: evaluate how well different multi-view SSL objectives capture each PID atom type.

### PID atom mapping (pid_id 0–9)
- 0–2: Unique (U1, U2, U3)
- 3–6: Redundancy (R12, R13, R23, R123)
- 7–9: Directional Synergy (S12→3, S13→2, S23→1)

### Key files
- `pid_sar3_dataset.py` — data generator (`PIDSar3DatasetGenerator`, `PIDDatasetConfig`)
- `pid_sar3_ssl.py` — SSL models and objectives
- `tests/test_pid_sar3_ssl_fused_confusions.py` (3546 lines) — main test suite
- `docs/` — written-up result reports

### SSL objectives compared
- **A**: 3× unimodal SimCLR (separate per-modality)
- **B**: pairwise InfoNCE sum (NT-Xent on all view pairs)
- **C**: TRIANGLE exact (area-based symmetric contrastive)
- **D**: ConFu (pairwise + trainable fused-pair-to-third term)

### Data modes
- `single_atom` (default) — each sample has exactly one PID atom, +noise
- `multi_atom` — each sample sums 5 active PID atoms + shared backbone + noise

### Shared backbone
`shared_backbone_gain` adds a per-sample random latent projected through a
*fixed* shared matrix (same for all samples within a view). With gain=4.0 and
`shared_backbone_tied_projection=True` (same P_shared across views), this term
strongly correlates all three views of every sample, making contrastive alignment easy.

---

## Where we left off (as of last session)

### Problem: overfitting in multi_atom / compositional regime
When using `_data_cfg_compositional_very_easy` (sigma=0.02, rho=0.8, hop=1,
multi_atom with 5 atoms/sample, shared_backbone_gain=4.0), joint SSL objectives
(B, C, D) overfit heavily on a finite 10k/2k train/val split:
- training loss drops fast (especially in epoch 1–5)
- validation loss rises after a few epochs → large val-train gap

### Investigation: `test_l0_optimization_gap_ablations_fixed_budget`
Located at lines 3393–3546 of `test_pid_sar3_ssl_fused_confusions.py`.
Results in: `test_outputs/pid_sar3_ssl_fused_confusions/l0_optimization_gap_ablations_*.{csv,png}`

#### Ablation variants tested
| Variant | shared_backbone_gain | weight_decay | epochs |
|---------|---------------------|--------------|--------|
| baseline | 4.0 | 1e-5 | 50 |
| lower_shared_gain | 1.0 | 1e-5 | 50 |
| strong_weight_decay | 4.0 | 1e-3 | 50 |
| lower_shared_plus_wd | 1.0 | 1e-3 | 50 |
| early_cap_10 | 4.0 | 1e-5 | 10 |

#### Key result: overfit drift (val_last - val_best), lower is better
| Variant | Pairwise InfoNCE | TRIANGLE | ConFu | Mean |
|---------|---------|---------|-------|------|
| baseline | 0.741 | 0.377 | 0.766 | 0.628 |
| lower_shared_gain | 0.866 | 0.171 | 0.849 | 0.629 |
| strong_weight_decay | 0.661 | 0.128 | 0.533 | 0.441 |
| lower_shared_plus_wd | 0.550 | 0.163 | 0.588 | 0.434 |
| early_cap_10 | 0.286 | 0.288 | 0.360 | 0.311 |

#### Conclusions from ablation
1. Lowering shared_backbone_gain alone does NOT reduce overfitting (even makes pairwise worse)
2. Strong weight decay (1e-3) helps all objectives
3. Early stopping at epoch 10 is most effective
4. Recommended anti-overfitting recipe: max_epochs=10 + val-loss checkpoint selection

---

## Probe impact study: `test_l0_overfitting_probe_impact` (DONE)

Test added at line 3621. Artifacts:
- `test_outputs/.../l0_overfitting_probe_impact.csv`
- `test_outputs/.../l0_overfitting_probe_impact.png`
- `test_outputs/.../l0_overfitting_probe_impact_curves.png`

### Results (10k train / 2k probe, compositional_very_easy, 50 epochs, cpu)

| Model | best_epoch | overfit_drift | pid10 best_val | pid10 final | fam3 best_val | fam3 final |
|-------|-----------|---------------|---------------|-------------|--------------|------------|
| pairwise_infonce | 3 | 0.560 | 9.3% | **10.8%** | **34.6%** | 34.1% |
| triangle_exact   | 11 | 0.062 | 8.8% | 9.1% | **37.0%** | 36.7% |
| confu_style      | 2 | 0.624 | 9.7% | **10.0%** | **35.8%** | 35.0% |

### Conclusions

1. **Overfitting does NOT hurt downstream performance** — final-epoch checkpoint is
   marginally better than best-val on pid10_acc in most cases. Answer to open question 1: no.

2. **All probe accuracies are near random** — 10-class random = 10%, 3-class = 33%.
   None of the objectives learn PID structure with fixed 10k data in the compositional regime.
   This is a **more fundamental problem than overfitting**.

3. **best_epoch is extremely early** (2–3 for pairwise/confu, 11 for triangle).
   Val loss spikes quickly but the "early good" checkpoint has no better representations.

4. **Root cause hypothesis**: The SSL objectives cannot learn the PID decomposition
   structure from a finite, pre-generated 10k dataset in the multi_atom regime.
   They need streaming/infinite data (as in the generator-based tests that do work).
   The val-loss signal is misleading — it measures contrastive loss memorization, not
   representation quality.

---

## Per-family and compositional ladder: `test_l0_family_*` + `test_l0_compositional_*` (DONE)

Six experiments (family-restricted single_atom + compositional ladder).
4/6 completed; multi_atom_2 and multi_atom_5 not yet run.
Aggregation test: `test_l0_aggregate_family_compositional_results` → writes MD.

### Results (probe_acc, best-val checkpoint, frozen linear probe on [h1,h2,h3])

| Experiment | n_cls | Random | A: Unimodal | B: Pairwise | C: Triangle | D: ConFu | Best (gap) |
|---|---|---|---|---|---|---|---|
| unique_only      | 3  | 33.3% | **41.1%** | 38.2% | 37.8% | 39.7% | A (+7.8pp) |
| redundancy_only  | 4  | 25.0% | 27.8% | 26.6% | 28.6% | **30.1%** | D (+5.1pp) |
| synergy_only     | 3  | 33.3% | 35.2% | 34.1% | **36.8%** | 36.7% | C (+3.4pp) |
| single_atom_all10| 10 | 10.0% | 10.8% | 11.7% | **12.4%** | 11.1% | C (+2.4pp) |

### Key conclusions

1. **Root cause identified: missing augmentation in joint fixed-data training.**
   - Joint objectives (B, C, D) peak at best_epoch=2–6 then collapse (drifts +0.6 to +0.9).
   - Model A (unimodal, uses VectorAugmenter) stays stable through epoch 34–49.
   - Without augmentation, joint objectives memorise specific sample identities rather
     than learning PID structure. After 2–6 epochs the contrastive loss has memorised
     the fixed (x1,x2,x3) pairs and the representations overfit to sample identity.

2. **Composition is NOT the root cause.**
   - single_atom_all10 is also near-random (+2.4pp best for 10-class).
   - Earlier hypothesis "multi_atom complexity causes failure" is WRONG.
   - The failure is the same regime regardless of composition_mode.

3. **The fix: add augmentation to joint fixed-data training loop.**
   - Model A already uses VectorAugmenter (jitter_std=0.08, feat_drop=0.08, gain 0.92-1.08).
   - Adding the same augmentation to `_compute_trimodal_ssl_loss` or its callers
     should stabilise joint model training dramatically.

4. **Family restriction only modestly helps** — in the latest rerun:
   unique-only peaks at `41.1%` (A), redundancy-only at `30.1%` (D), synergy-only at `36.8%` (C);
   all are only a few points above random, so the fixed-data memorisation problem remains.

---

## Open questions / next steps

1. ~~Does overfitting hurt downstream probe performance?~~ **ANSWERED: No (probe impact study).**

2. ~~Is the failure specific to multi_atom?~~ **ANSWERED: No (single_atom_all10 also fails).**

3. **[PRIORITY] Add augmentation to joint fixed-data training.**
   Modify `_train_trimodal_objective_fixed_dataset_best_val` (and the new
   `_train_trimodal_both_checkpoints`) to apply VectorAugmenter to x1/x2/x3 before
   computing the SSL loss. Re-run the full ladder and expect major improvements.

4. **Run multi_atom_2 and multi_atom_5** experiments (they didn't complete).
   These will fill in the compositional part of the results table, but we already
   expect near-random performance consistent with single_atom_all10.

5. **Streaming vs fixed-data at matched budget**: confirm that streaming training
   at ~78 steps × 128 ≈ 10k samples does work (hypothesis: yes, because generator
   provides fresh samples each step = implicit augmentation).
