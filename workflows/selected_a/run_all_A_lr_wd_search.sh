#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-quick}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/test_outputs/v3_runs_A_lrwd_search}"
SEED="${SEED:-101}"

if [[ -z "${DEVICE:-}" ]]; then
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

case "$MODE" in
  quick)
    N_TRAIN=4000
    N_PROBE_TRAIN=1200
    N_PROBE_TEST=400
    PRETRAIN_EPOCHS=20
    PROBE_EPOCHS=300
    BATCH_SIZE=256
    ;;
  standard)
    N_TRAIN=12000
    N_PROBE_TRAIN=3000
    N_PROBE_TEST=1000
    PRETRAIN_EPOCHS=60
    PROBE_EPOCHS=300
    BATCH_SIZE=512
    ;;
  full)
    N_TRAIN=50000
    N_PROBE_TRAIN=5000
    N_PROBE_TEST=1000
    PRETRAIN_EPOCHS=200
    PROBE_EPOCHS=300
    BATCH_SIZE=512
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [quick|standard|full]"
    exit 1
    ;;
esac

CONFIGS=(A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14)
METHODS=(simclr pairwise_nce triangle confu masked_raw masked_emb comm infmask)
LRS=(1e-4 3e-4 1e-3 3e-3)
WDS=(0.0 1e-5 1e-4 1e-3)

mkdir -p "$OUT_ROOT"

echo "Mode: $MODE"
echo "Device: $DEVICE"
echo "Python: $PYTHON_BIN"
echo "Output root: $OUT_ROOT"
echo "Seed: $SEED"
echo "Grid size per method/config: ${#LRS[@]}x${#WDS[@]} = $((${#LRS[@]} * ${#WDS[@]}))"
echo "Probe epochs: $PROBE_EPOCHS"

print_section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

print_run_header() {
  echo
  echo "------------------------------------------------------------"
  echo "$1"
  echo "------------------------------------------------------------"
}

run_cmd() {
  echo "cmd: $1 ..."
  "$@"
}

manifest="$OUT_ROOT/search_manifest.csv"
echo "config,method,lr,weight_decay,seed,run_dir" > "$manifest"

for config in "${CONFIGS[@]}"; do
  print_section "A-family sweep | config=$config"
  for method in "${METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
      for wd in "${WDS[@]}"; do
        lr_tag="${lr//./p}"
        wd_tag="${wd//./p}"
        run_dir="$OUT_ROOT/$config/$method/lr_${lr_tag}__wd_${wd_tag}/seed_${SEED}"
        mkdir -p "$run_dir"
        echo "$config,$method,$lr,$wd,$SEED,$run_dir" >> "$manifest"

        print_run_header "PRETRAIN+PROBE | config=$config | method=$method | lr=$lr | wd=$wd | seed=$SEED"
        echo "run_dir: $run_dir"

        run_cmd "$PYTHON_BIN" train_pretrain.py \
          --method "$method" \
          --config "$config" \
          --Q 7 \
          --D 44 \
          --D_info 4 \
          --n_train "$N_TRAIN" \
          --d_model 64 \
          --d_z 64 \
          --n_layers 2 \
          --tau 0.07 \
          --lambda_contr 1.0 \
          --batch_size "$BATCH_SIZE" \
          --epochs "$PRETRAIN_EPOCHS" \
          --lr "$lr" \
          --weight_decay "$wd" \
          --mask_ratio 0.5 \
          --ema_momentum 0.996 \
          --masked_emb_var_weight 1.0 \
          --lambda_mask 1.0 \
          --n_mask_samples 1 \
          --seed "$SEED" \
          --device "$DEVICE" \
          --save_dir "$run_dir/pretrain"

        run_cmd "$PYTHON_BIN" train_probe.py \
          --checkpoint "$run_dir/pretrain/final.pt" \
          --probe_config "$config" \
          --Q 7 \
          --D 44 \
          --n_probe_train "$N_PROBE_TRAIN" \
          --n_probe_test "$N_PROBE_TEST" \
          --probe_epochs "$PROBE_EPOCHS" \
          --device "$DEVICE" \
          --save_dir "$run_dir/probe"
      done
    done
  done
done

echo
echo "Finished A-family LR/WD search."
echo "Manifest: $manifest"
echo "Outputs:  $OUT_ROOT"
