#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-101}"

OUT_ROOT="${OUT_ROOT:-$ROOT/test_outputs/v3_runs_selected_search}"
CONFIGS=(A1 A4 A7 A8 A11 A12)
METHODS=(simclr pairwise_nce triangle confu masked_raw masked_emb comm infmask)

LRS=(1e-4 3e-4 1e-3 3e-3)
WDS=(0.0 1e-5 1e-4 1e-3)

N_TRAIN="${N_TRAIN:-12000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-512}"

N_PROBE_TRAIN="${N_PROBE_TRAIN:-3000}"
N_PROBE_TEST="${N_PROBE_TEST:-1000}"
PROBE_EPOCHS="${PROBE_EPOCHS:-300}"
NONLINEAR_HIDDEN="${NONLINEAR_HIDDEN:-256}"

mkdir -p "$OUT_ROOT"

run_one() {
  local config="$1"
  local method="$2"
  local lr="$3"
  local wd="$4"
  local tau="$5"
  local view_noise_std="$6"
  local triangle_alpha="$7"
  local confu_fuse_weight="$8"
  local mask_ratio="$9"
  local ema_momentum="${10}"
  local masked_emb_var_weight="${11}"
  local lambda_mask="${12}"
  local n_mask_samples="${13}"

  local hp_tag="lr_${lr//./p}__wd_${wd//./p}__tau_${tau//./p}__noise_${view_noise_std//./p}__cfw_${confu_fuse_weight//./p}__mr_${mask_ratio//./p}__ema_${ema_momentum//./p}__var_${masked_emb_var_weight//./p}__kms_${n_mask_samples}"
  local run_dir="$OUT_ROOT/$config/$method/$hp_tag/seed_${SEED}"

  echo
  echo "============================================================"
  echo "config=$config method=$method hp=$hp_tag"
  echo "============================================================"

  "$PYTHON_BIN" train_pretrain.py \
    --method "$method" \
    --config "$config" \
    --Q 7 \
    --D 44 \
    --D_info 4 \
    --n_train "$N_TRAIN" \
    --d_model 64 \
    --d_z 64 \
    --n_layers 2 \
    --tau "$tau" \
    --lambda_contr 1.0 \
    --lambda_mask "$lambda_mask" \
    --n_mask_samples "$n_mask_samples" \
    --mask_ratio "$mask_ratio" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$PRETRAIN_EPOCHS" \
    --lr "$lr" \
    --weight_decay "$wd" \
    --view_noise_std "$view_noise_std" \
    --triangle_alpha "$triangle_alpha" \
    --confu_fuse_weight "$confu_fuse_weight" \
    --ema_momentum "$ema_momentum" \
    --masked_emb_var_weight "$masked_emb_var_weight" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --save_dir "$run_dir/pretrain"

  "$PYTHON_BIN" train_probe.py \
    --checkpoint "$run_dir/pretrain/final.pt" \
    --probe_config "$config" \
    --Q 7 \
    --D 44 \
    --n_probe_train "$N_PROBE_TRAIN" \
    --n_probe_test "$N_PROBE_TEST" \
    --probe_epochs "$PROBE_EPOCHS" \
    --device "$DEVICE" \
    --save_dir "$run_dir/probe_linear"

  "$PYTHON_BIN" run_nonlinear_probe.py \
    --checkpoint "$run_dir/pretrain/final.pt" \
    --probe_config "$config" \
    --Q 7 \
    --D 44 \
    --D_info 4 \
    --n_probe_train "$N_PROBE_TRAIN" \
    --n_probe_test "$N_PROBE_TEST" \
    --probe_epochs "$PROBE_EPOCHS" \
    --hidden_dim "$NONLINEAR_HIDDEN" \
    --device "$DEVICE" \
    --save_dir "$run_dir/probe_nonlinear"
}

for config in "${CONFIGS[@]}"; do
  for method in "${METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
      for wd in "${WDS[@]}"; do
        taus=("0.07")
        view_noises=("0.1")
        confu_weights=("0.5")
        mask_ratios=("0.5")
        ema_momentums=("0.996")
        var_weights=("1.0")
        mask_samples=("1")
        triangle_alpha="0.0"
        lambda_mask="1.0"

        case "$method" in
          simclr)
            taus=("0.07" "0.1" "0.2" "0.5")
            view_noises=("0.05" "0.1" "0.2")
            ;;
          pairwise_nce)
            taus=("0.07" "0.1" "0.2" "0.4")
            ;;
          triangle)
            taus=("0.07" "0.1" "0.2" "0.4")
            ;;
          confu)
            taus=("0.07" "0.1" "0.2" "0.4")
            confu_weights=("0.25" "0.5" "0.75")
            ;;
          comm)
            taus=("0.07" "0.1" "0.2")
            view_noises=("0.05" "0.1" "0.2")
            ;;
          infmask)
            taus=("0.07" "0.1" "0.2")
            view_noises=("0.05" "0.1" "0.2")
            mask_ratios=("0.3" "0.5" "0.7")
            mask_samples=("1" "2")
            ;;
          masked_raw)
            mask_ratios=("0.3" "0.5" "0.7")
            ;;
          masked_emb)
            mask_ratios=("0.3" "0.5" "0.7")
            ema_momentums=("0.99" "0.996" "0.999")
            var_weights=("0.25" "1.0" "4.0")
            ;;
        esac

        for tau in "${taus[@]}"; do
          for view_noise_std in "${view_noises[@]}"; do
            for confu_fuse_weight in "${confu_weights[@]}"; do
              for mask_ratio in "${mask_ratios[@]}"; do
                for ema_momentum in "${ema_momentums[@]}"; do
                  for masked_emb_var_weight in "${var_weights[@]}"; do
                    for n_mask_samples in "${mask_samples[@]}"; do
                      run_one \
                        "$config" "$method" "$lr" "$wd" "$tau" "$view_noise_std" \
                        "$triangle_alpha" "$confu_fuse_weight" "$mask_ratio" \
                        "$ema_momentum" "$masked_emb_var_weight" "$lambda_mask" "$n_mask_samples"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo
echo "Finished selected hyperparameter search with linear + nonlinear probing."
echo "Outputs: $OUT_ROOT"
