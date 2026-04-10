#!/bin/bash
# Weak-supervision (v20-style, no GT mask) with patch size 16^3 or 8^3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-7}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MINI_BATCH="${MINI_BATCH:-8}"
MAX_PATCHES="${MAX_PATCHES:-512}"
MASTER_PORT="${MASTER_PORT:-29612}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${REPO_ROOT}"

run_weak () {
  local patch=$1
  local run_name="${RUN_NAME:-weak_patch${patch}}"
  local launcher_log="${ARTIFACT_ROOT}/${run_name}/logs/launcher.log"
  mkdir -p "$(dirname "${launcher_log}")"

  echo "=========================================="
  echo "Weak supervision, patch_size=${patch}^3 (no resume)"
  echo "Run name: ${run_name}"
  echo "Log: ${launcher_log}"
  echo "=========================================="

  nohup "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    scripts/train_volumetric_ddp.py \
    --config configs/volumetric_v2_config.yaml \
    --patch_size "${patch}" \
    --weak_supervision \
    --no_resume \
    --batch_size "${BATCH_SIZE}" \
    --mini_batch_size "${MINI_BATCH}" \
    --max_patches "${MAX_PATCHES}" \
    --mixed_precision \
    --output_root "${ARTIFACT_ROOT}" \
    --run_name "${run_name}" \
    > "${launcher_log}" 2>&1 &

  echo "Started. PID=$!"
}

PATCH="${1:-16}"
shift || true
if [[ "${PATCH}" != "16" && "${PATCH}" != "8" ]]; then
  echo "Usage: $0 16|8 [extra args...]"
  exit 1
fi
run_weak "${PATCH}" "$@"