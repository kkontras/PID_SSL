# PID_SSL

`PID_SSL` is a synthetic multimodal learning repo for studying how self-supervised objectives recover different PID-style information patterns across three inputs `x1`, `x2`, and `x3`.

The repo is organized around three things:

- generating controlled synthetic datasets where the label is built from selected information atoms
- pretraining a 3-branch multimodal encoder with different SSL objectives
- evaluating the learned representation with frozen linear and nonlinear probes, then aggregating the best runs into final heatmaps

## 1. What This Repo Studies

Each sample has three modalities:

- `x1`
- `x2`
- `x3`

The target is generated from one or more selected information atoms. These atoms control whether the label is available uniquely in one modality, redundantly across modalities, synergistically across modalities, or in an asymmetric pair-redundant structure.

The maintained workflow in this repo focuses on:

- SSL pretraining on selected `A` configurations
- frozen linear probe evaluation
- frozen nonlinear probe evaluation
- aggregation into `test_outputs/aggregated_results`
- final heatmaps for a small selected set of `A` configs

## 2. Repo Layout

Core entrypoints at the repo root:

- `train_pretrain.py`: SSL pretraining
- `train_probe.py`: frozen linear probe evaluation
- `run_nonlinear_probe.py`: frozen nonlinear probe evaluation
- `train_e2e.py`: supervised baseline training
- `verify_dataset.py`: quick dataset sanity checks

Core code:

- `data/`: synthetic PID dataset generator
- `models/`: encoder, projection, fusion, decoder, and cross-modal transformer modules
- `losses/`: SSL objective implementations
- `probing/`: linear probe, nonlinear probe, retrieval, and per-atom evaluation
- `configs/`, `utils/`, `tests/`: support code and tests

Maintained experiment workflow:

- `workflows/selected_a/`: hyperparameter search, aggregation, best-run export, and selected-A heatmap scripts

Legacy scripts kept for reference only:

- `scripts/legacy/`

## 3. Dataset Configs And Atoms

The dataset definitions live in [`data/dataset_v3.py`](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/PID_SSL/data/dataset_v3.py).

### Single-atom A configs

These are the main configs used in the presentable workflow.

- `A1`: `uniq_1` — label information exists only in `x1`
- `A2`: `uniq_2` — label information exists only in `x2`
- `A3`: `uniq_3` — label information exists only in `x3`
- `A4`: `red_12` — the same informative variable is present in `x1` and `x2`
- `A5`: `red_13` — the same informative variable is present in `x1` and `x3`
- `A6`: `red_23` — the same informative variable is present in `x2` and `x3`
- `A7`: `red_123` — the same informative variable is shared by all three modalities
- `A8`: `syn_12` — the target depends jointly on `x1` and `x2`; neither alone is sufficient
- `A9`: `syn_13` — synergy between `x1` and `x3`
- `A10`: `syn_23` — synergy between `x2` and `x3`
- `A11`: `syn_123` — the target depends jointly on `x1`, `x2`, and `x3`
- `A12`: `pairred_12_3` — `x1` and `x2` form a pair that predicts information carried in `x3`
- `A13`: `pairred_13_2` — `x1` and `x3` predict information carried in `x2`
- `A14`: `pairred_23_1` — `x2` and `x3` predict information carried in `x1`

### Intuition for the atom families

- `uniq_*`: one modality alone contains the needed label information
- `red_*`: the same signal is copied across two or three modalities
- `syn_*`: useful information is split across modalities and must be combined
- `pairred_*`: a structured asymmetric case where a modality is predictable from a pair

### Other config groups

The code also defines:

- `B*`: multi-atom mixtures
- `C*`: asymmetric or stress-test mixtures

The current presentable pipeline is centered on the `A*` configs.

## 4. SSL Methods

The SSL objective dispatch lives in [`losses/combined.py`](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/PID_SSL/losses/combined.py).

Available methods for `train_pretrain.py`:

- `simclr`: creates two noisy views of the same 3-modal sample and applies SimCLR-style contrastive learning independently per modality
- `pairwise_nce`: directly aligns modality pairs with pairwise InfoNCE losses
- `triangle`: extends pairwise contrast with an extra triangle consistency term across the three modalities
- `confu`: learns pairwise fused representations (`f12`, `f13`, `f23`) and applies contrastive learning on both unimodal and fused branches
- `comm`: builds a single fused embedding for all three modalities and contrasts both unimodal and fused views
- `infmask`: extends `comm` by adding masked-view consistency of the fused representation
- `masked_raw`: masks raw input features and trains the model to reconstruct the masked raw inputs from latent representations
- `masked_emb`: masks inputs and trains a student encoder to predict teacher embeddings, with an EMA teacher
- `none`: no SSL objective; mainly useful as a control path in the training code

### Quick method intuition

- If you care about strong pairwise alignment baselines, start with `pairwise_nce`.
- If you want classic augmentation-based contrastive learning, use `simclr`.
- If you want explicit fusion modules for cross-modal interactions, use `confu` or `comm`.
- If you want masking-based objectives, use `masked_raw`, `masked_emb`, or `infmask`.
- If you expect synergy-heavy tasks, methods that explicitly combine modalities are usually the most relevant ones to compare.

## 5. Environment

The Conda environment file for this repo is [`environment.yml`](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/PID_SSL/environment.yml).

Create the environment with:

```bash
conda env create -f environment.yml
conda activate pid_ssl
```

If you need to update an existing env instead:

```bash
conda env update -f environment.yml --prune
conda activate pid_ssl
```

Notes:

- `environment.yml` was exported from `/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new` with `--from-history`
- this is now the only environment YAML kept in the repo
- because it is a `--from-history` export, it may be cleaner than a full lock file but not as exhaustive

Main libraries used by the code are PyTorch, NumPy, and Matplotlib.

## 6. Quick Start

### 6.1 Pretrain one model

Example: pretrain `pairwise_nce` on `A8`.

```bash
python train_pretrain.py \
  --method pairwise_nce \
  --config A8 \
  --Q 7 \
  --D 44 \
  --D_info 4 \
  --n_train 12000 \
  --d_model 64 \
  --d_z 64 \
  --n_layers 2 \
  --tau 0.07 \
  --lambda_contr 1.0 \
  --batch_size 512 \
  --epochs 60 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 101 \
  --device cuda \
  --save_dir test_outputs/example_a8_pairwise/pretrain
```

This writes a checkpoint and training curves under the `save_dir`, including `final.pt`.

### 6.2 Run a frozen linear probe

```bash
python train_probe.py \
  --checkpoint test_outputs/example_a8_pairwise/pretrain/final.pt \
  --probe_config A8 \
  --Q 7 \
  --D 44 \
  --n_probe_train 3000 \
  --n_probe_test 1000 \
  --probe_epochs 300 \
  --device cuda \
  --save_dir test_outputs/example_a8_pairwise/probe_linear
```

This writes:

- `probe_results.json`
- `probe_summary.csv`
- `retrieval_summary.csv`
- probe history and curve files

### 6.3 Run a frozen nonlinear probe

```bash
python run_nonlinear_probe.py \
  --checkpoint test_outputs/example_a8_pairwise/pretrain/final.pt \
  --probe_config A8 \
  --Q 7 \
  --D 44 \
  --D_info 4 \
  --n_probe_train 3000 \
  --n_probe_test 1000 \
  --probe_epochs 300 \
  --hidden_dim 256 \
  --device cuda \
  --save_dir test_outputs/example_a8_pairwise/probe_nonlinear
```

This writes `nonlinear_probe_results.json` and nonlinear probe history/curve artifacts.

## 7. How To Choose A Config

Use these as a practical starting point:

- Start with `A1`, `A4`, `A8`, `A11`, `A12` if you want one representative from each major regime.
- Use `A8` and `A11` if your focus is synergy.
- Use `A12` if you want the asymmetric pair-redundant case.
- Use `A1` or `A4` as simpler sanity checks before running harder synergy experiments.

## 8. Selected-A Workflow

The maintained end-to-end workflow lives in [`workflows/selected_a/README.md`](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/PID_SSL/workflows/selected_a/README.md).

The practical flow is:

1. Run the coarse LR/WD search.
2. Select one hyperparameter family per method/family.
3. Rerun selected configs with both linear and nonlinear probes.
4. Aggregate all discovered runs into `test_outputs/aggregated_results`.
5. Export the best runs and make final heatmaps.

### 8.1 Coarse LR/WD search over A families

```bash
bash workflows/selected_a/run_all_A_lr_wd_search.sh standard
```

Modes:

- `quick`
- `standard`
- `full`

You can override common settings through environment variables, for example:

```bash
DEVICE=cuda PYTHON_BIN=python SEED=101 bash workflows/selected_a/run_all_A_lr_wd_search.sh standard
```

### 8.2 Select best hyperparameter families

This script consumes a manifest CSV and selects the best family-level setting by method.

```bash
python workflows/selected_a/select_a_family_hparams.py \
  --manifest test_outputs/some_manifest.csv \
  --out_prefix test_outputs/selected_hparams/a_family
```

### 8.3 Run selected search with both probes

```bash
bash workflows/selected_a/selected_search_with_both_probes.sh
```

This runs:

- pretraining
- frozen linear probe
- frozen nonlinear probe

for the selected config set used in the final presentation workflow.

### 8.4 Expanded method-specific search for selected configs

```bash
bash workflows/selected_a/run_expanded_method_hparam_search_selected.sh
```

This expands method-specific hyperparameters for selected configs when the base search is not yet strong enough.

### 8.5 Build the aggregated results tree

```bash
python workflows/selected_a/build_aggregated_results.py
```

This collects runs from multiple experiment roots and assembles them under:

- `test_outputs/aggregated_results/`

### 8.6 Export best runs and create final heatmaps

```bash
python workflows/selected_a/export_selected_best_runs_csv.py
python workflows/selected_a/plot_selected_linear_unified_heatmap.py
python workflows/selected_a/plot_selected_nonlinear_unified_heatmap.py
```

These produce the final selected-A summary CSVs and heatmaps in `test_outputs/`.

## 9. Supervised Baseline

The repo also includes a supervised baseline path via `train_e2e.py`. Use this when you want an upper-bound comparison against the SSL pipelines.

## 10. Output Conventions

Most runs write under `test_outputs/`.

Common subfolders and files:

- `pretrain/final.pt`: pretrained checkpoint
- `pretrain/history.csv`: pretraining loss history
- `probe_linear/probe_results.json`: linear probe score
- `probe_linear/retrieval_summary.csv`: retrieval metrics
- `probe_nonlinear/nonlinear_probe_results.json`: nonlinear probe score
- `aggregated_results/`: merged run tree used by the final plotting scripts

## 11. Minimal Repro Example

If you just want one complete SSL run plus both probes:

```bash
python train_pretrain.py \
  --method confu \
  --config A8 \
  --n_train 12000 \
  --epochs 60 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --device cuda \
  --seed 101 \
  --save_dir test_outputs/tutorial_confu_a8/pretrain

python train_probe.py \
  --checkpoint test_outputs/tutorial_confu_a8/pretrain/final.pt \
  --probe_config A8 \
  --n_probe_train 3000 \
  --n_probe_test 1000 \
  --probe_epochs 300 \
  --device cuda \
  --save_dir test_outputs/tutorial_confu_a8/probe_linear

python run_nonlinear_probe.py \
  --checkpoint test_outputs/tutorial_confu_a8/pretrain/final.pt \
  --probe_config A8 \
  --n_probe_train 3000 \
  --n_probe_test 1000 \
  --probe_epochs 300 \
  --hidden_dim 256 \
  --device cuda \
  --save_dir test_outputs/tutorial_confu_a8/probe_nonlinear
```

## 12. Sanity Checks

Useful checks while developing:

- run `python verify_dataset.py` to inspect dataset generation
- use a smaller `--n_train` and fewer `--epochs` for smoke tests
- start on `A1` or `A4` before moving to `A8` or `A11`
- compare linear and nonlinear probes to see whether the representation is linearly accessible or only recoverable with extra probe capacity

## 13. Notes

- The synthetic label cardinality scales as `Q^(number of active atoms)`.
- `A*` configs are single-atom datasets and are the easiest place to interpret behavior cleanly.
- `masked_emb` uses an EMA teacher.
- `confu` and `comm` explicitly include learned multimodal fusion in the objective.
- the selected-A pipeline is the maintained, presentable path in this repo
