#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
RUN_NAME="${RUN_NAME:-fullsup_seg}"
PYTHON_BIN="${PYTHON_BIN:-/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/bin/python}"
LAUNCHER_LOG="${ARTIFACT_ROOT}/${RUN_NAME}/logs/launcher.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/lib/python3.11/site-packages/torch/lib}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$(dirname "${LAUNCHER_LOG}")"
cd "${REPO_ROOT}"

nohup "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-7}" \
  --master_port="${MASTER_PORT:-29624}" \
  scripts/train_fullsup_seg_ddp.py \
  --config configs/fullsup_seg.yaml \
  --output_root "${ARTIFACT_ROOT}" \
  --run_name "${RUN_NAME}" \
  "$@" > "${LAUNCHER_LOG}" 2>&1 & disown

echo "Started fullsup_seg."
echo "launcher log: ${LAUNCHER_LOG}"