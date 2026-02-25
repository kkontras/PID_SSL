#!/usr/bin/env bash
set -euo pipefail

# Runs the three fixed-data family-only probe experiments:
# - unique_only
# - redundancy_only
# - synergy_only
#
# Outputs are written under:
#   test_outputs/pid_sar3_ssl_fused_confusions/
#
# Optional env overrides (examples):
#   PIDSSL_EXP_DEVICE=cpu
#   PIDSSL_EXP_EPOCHS=50
#   PIDSSL_EXP_N_PER_PID=3000
#
# Usage:
#   bash run_l0_family_only_probes.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_FILE="tests/test_pid_sar3_ssl_fused_confusions.py"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} not found"
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1; then
  echo "Error: pytest is not installed for ${PYTHON_BIN}"
  echo "Install it with:"
  echo "  ${PYTHON_BIN} -m pip install pytest"
  exit 1
fi

echo "[1/4] Running unique_only family probe..."
"${PYTHON_BIN}" -m pytest -q "${TEST_FILE}::test_l0_family_unique_probe" -s

echo "[2/4] Running redundancy_only family probe..."
"${PYTHON_BIN}" -m pytest -q "${TEST_FILE}::test_l0_family_redundancy_probe" -s

echo "[3/4] Running synergy_only family probe..."
"${PYTHON_BIN}" -m pytest -q "${TEST_FILE}::test_l0_family_synergy_probe" -s

echo "[4/4] Aggregating family/compositional results table..."
"${PYTHON_BIN}" -m pytest -q "${TEST_FILE}::test_l0_aggregate_family_compositional_results" -s

echo
echo "Done. Key outputs:"
echo "  - test_outputs/pid_sar3_ssl_fused_confusions/l0_exp_unique_only.csv"
echo "  - test_outputs/pid_sar3_ssl_fused_confusions/l0_exp_redundancy_only.csv"
echo "  - test_outputs/pid_sar3_ssl_fused_confusions/l0_exp_synergy_only.csv"
echo "  - test_outputs/pid_sar3_ssl_fused_confusions/l0_family_compositional_results.md"
