#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
THRESHOLD="${THRESHOLD:-0.95}"
OVERWRITE="${OVERWRITE:-0}"

echo "Device: $DEVICE"
echo "Python: $PYTHON_BIN"
echo "Threshold: $THRESHOLD"
echo "Output root: $ROOT_DIR/test_outputs/expanded_method_hparam_search_selected"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/run_expanded_method_hparam_search_selected.py"
  --python_bin "$PYTHON_BIN"
  --device "$DEVICE"
  --threshold "$THRESHOLD"
)

if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

printf 'cmd: %s ...\n' "${cmd[0]} ${cmd[1]}"
"${cmd[@]}"

echo
echo "To remake the plot:"
echo "python workflows/selected_a/plot_expanded_method_hparam_search_selected_heatmap.py"
