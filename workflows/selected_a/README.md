# Selected-A Workflow

This folder contains the scripts that drive the presentable results pipeline for:

- hyperparameter search over SSL pretraining plus linear and nonlinear probes
- aggregation into `test_outputs/aggregated_results`
- best-run selection
- final linear/nonlinear heatmaps for the selected A-configs

Suggested order:

1. `run_all_A_lr_wd_search.sh`
2. `select_a_family_hparams.py`
3. `selected_search_with_both_probes.sh` or `run_expanded_method_hparam_search_selected.sh`
4. `build_aggregated_results.py`
5. `export_selected_best_runs_csv.py`
6. `plot_selected_linear_unified_heatmap.py`
7. `plot_selected_nonlinear_unified_heatmap.py`

Core training entrypoints remain at the repo root: `train_pretrain.py`, `train_probe.py`, `run_nonlinear_probe.py`, and `train_e2e.py`.
