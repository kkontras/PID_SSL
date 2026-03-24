#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-quick}"
STAGE="${2:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/test_outputs/v3_runs}"

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
    VERIFY_SAMPLES=4000
    N_TRAIN=4000
    N_PROBE_TRAIN=1200
    N_PROBE_TEST=400
    PRETRAIN_EPOCHS=20
    PROBE_EPOCHS=60
    E2E_EPOCHS=20
    BATCH_SIZE=256
    ;;
  standard)
    VERIFY_SAMPLES=12000
    N_TRAIN=12000
    N_PROBE_TRAIN=3000
    N_PROBE_TEST=1000
    PRETRAIN_EPOCHS=60
    PROBE_EPOCHS=100
    E2E_EPOCHS=60
    BATCH_SIZE=512
    ;;
  full)
    VERIFY_SAMPLES=20000
    N_TRAIN=50000
    N_PROBE_TRAIN=5000
    N_PROBE_TEST=1000
    PRETRAIN_EPOCHS=200
    PROBE_EPOCHS=100
    E2E_EPOCHS=200
    BATCH_SIZE=512
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [quick|standard|full] [all|sanity|single_atom|multi_atom|stress|e2e]"
    exit 1
    ;;
esac

mkdir -p "$OUT_ROOT"

echo "Mode: $MODE"
echo "Stage: $STAGE"
echo "Device: $DEVICE"
echo "Python: $PYTHON_BIN"
echo "Output root: $OUT_ROOT"

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
  echo
  echo "cmd: $1 ..."
  "$@"
}

run_verify() {
  local tag="$1"
  shift
  run_cmd "$PYTHON_BIN" verify_dataset.py \
    --configs "$@" \
    --Q 7 \
    --D 44 \
    --D_info 4 \
    --n_samples "$VERIFY_SAMPLES" \
    --device "$DEVICE" \
    --sigma_info 0.002 \
    --mu_bg 0.40 \
    --sigma_bg 0.10
}

run_pretrain_probe() {
  local method="$1"
  local config="$2"
  local seed="$3"
  local run_dir="$OUT_ROOT/$config/$method/seed_$seed"

  mkdir -p "$run_dir"

  print_run_header "PRETRAIN+PROBE | config=$config | method=$method | seed=$seed"
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
    --lr 1e-3 \
    --seed "$seed" \
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
}

run_e2e() {
  local method="$1"
  local config="$2"
  local seed="$3"
  local run_dir="$OUT_ROOT/e2e/$config/$method/seed_$seed"

  mkdir -p "$run_dir"

  print_run_header "E2E | config=$config | method=$method | seed=$seed"
  echo "run_dir: $run_dir"

  run_cmd "$PYTHON_BIN" train_e2e.py \
    --method "$method" \
    --config "$config" \
    --Q 7 \
    --D 44 \
    --D_info 4 \
    --n_train "$N_TRAIN" \
    --lambda_contr 0.1 \
    --tau 0.07 \
    --batch_size "$BATCH_SIZE" \
    --epochs "$E2E_EPOCHS" \
    --lr 1e-3 \
    --device "$DEVICE" \
    --save_dir "$run_dir"
}

stage_sanity() {
  print_section "Stage 0: dataset sanity checks"
  run_verify "single_atom" A1 A4 A8 A11
  if [[ "$MODE" == "quick" ]]; then
    run_verify "composed" B10 C3
  else
    run_verify "composed" B4 B10 C2 C3
  fi
}

stage_single_atom() {
  print_section "Stage 1: single-atom pilots"
  local configs=(A1 A4 A8)
  local methods=(simclr pairwise_nce triangle confu comm infmask)
  local seed=101
  local config method
  for config in "${configs[@]}"; do
    for method in "${methods[@]}"; do
      run_pretrain_probe "$method" "$config" "$seed"
    done
  done
}

stage_multi_atom() {
  print_section "Stage 2: mixed multi-atom benchmarks"
  local configs
  if [[ "$MODE" == "quick" ]]; then
    configs=(B10)
  else
    configs=(B4 B10)
  fi
  local methods=(simclr pairwise_nce triangle confu comm infmask)
  local seed=202
  local config method
  for config in "${configs[@]}"; do
    for method in "${methods[@]}"; do
      run_pretrain_probe "$method" "$config" "$seed"
    done
  done
}

stage_stress() {
  print_section "Stage 3: asymmetric stress tests"
  local configs
  if [[ "$MODE" == "quick" ]]; then
    configs=(C3)
  else
    configs=(C2 C3)
  fi
  local methods=(pairwise_nce triangle confu comm infmask)
  local seed=303
  local config method
  for config in "${configs[@]}"; do
    for method in "${methods[@]}"; do
      run_pretrain_probe "$method" "$config" "$seed"
    done
  done
}

stage_e2e() {
  print_section "Stage 4: end-to-end baselines on representative configs"
  local pairs
  if [[ "$MODE" == "quick" ]]; then
    pairs=(
      "none B10"
      "simclr B10"
      "pairwise_nce B10"
      "triangle B10"
      "confu B10"
      "pairwise_nce C3"
      "triangle C3"
      "confu C3"
    )
  else
    pairs=(
      "none B4"
      "simclr B4"
      "pairwise_nce B4"
      "triangle B4"
      "confu B4"
      "pairwise_nce C2"
      "triangle C2"
      "confu C2"
    )
  fi
  local seed=404
  local pair method config
  for pair in "${pairs[@]}"; do
    read -r method config <<<"$pair"
    run_e2e "$method" "$config" "$seed"
  done
}

case "$STAGE" in
  all)
    stage_sanity
    stage_single_atom
    stage_multi_atom
    stage_stress
    stage_e2e
    ;;
  sanity)
    stage_sanity
    ;;
  single_atom)
    stage_single_atom
    ;;
  multi_atom)
    stage_multi_atom
    ;;
  stress)
    stage_stress
    ;;
  e2e)
    stage_e2e
    ;;
  *)
    echo "Unknown stage: $STAGE"
    echo "Usage: $0 [quick|standard|full] [all|sanity|single_atom|multi_atom|stress|e2e]"
    exit 1
    ;;
esac

echo
echo "Finished V3 experiment ladder."
echo "Outputs are under: $OUT_ROOT"
