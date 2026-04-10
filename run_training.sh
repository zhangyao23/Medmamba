#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
RUN_NAME="${RUN_NAME:-volumetric_v2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

NUM_GPUS="${NUM_GPUS:-7}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MINI_BATCH="${MINI_BATCH:-8}"
MAX_PATCHES="${MAX_PATCHES:-64}"
MASTER_PORT="${MASTER_PORT:-29611}"
RESUME_CKPT="${RESUME_CKPT:-}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_ROOT}"

echo "=========================================="
echo "Training Launch"
echo "=========================================="
echo "  Repo root:    ${REPO_ROOT}"
echo "  Artifacts:    ${ARTIFACT_ROOT}"
echo "  Run name:     ${RUN_NAME}"
echo "  Python:       ${PYTHON_BIN}"
echo "  GPUs:         ${GPU_IDS} (${NUM_GPUS} total)"
echo "  Batch size:   ${BATCH_SIZE} x ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS))"
echo "=========================================="

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node="${NUM_GPUS}"
  --master_port="${MASTER_PORT}"
  scripts/train_volumetric_ddp.py
  --config configs/volumetric_v2_config.yaml
  --batch_size "${BATCH_SIZE}"
  --mini_batch_size "${MINI_BATCH}"
  --max_patches "${MAX_PATCHES}"
  --mixed_precision
  --output_root "${ARTIFACT_ROOT}"
  --run_name "${RUN_NAME}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  cmd+=(--resume "${RESUME_CKPT}")
fi

cmd+=("$@")
"${cmd[@]}"