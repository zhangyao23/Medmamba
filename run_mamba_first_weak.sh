#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
RUN_NAME="${RUN_NAME:-v20_mamba_first_weak}"
PYTHON_BIN="${PYTHON_BIN:-/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/bin/python}"
LAUNCHER_LOG="${ARTIFACT_ROOT}/${RUN_NAME}/logs/launcher.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4,7}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/lib/python3.11/site-packages/torch/lib}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$(dirname "${LAUNCHER_LOG}")"
cd "${REPO_ROOT}"

nohup "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-5}" \
  --master_port="${MASTER_PORT:-29612}" \
  scripts/train_volumetric_ddp.py \
  --config configs/v20_mamba_first_weak.yaml \
  --output_root "${ARTIFACT_ROOT}" \
  --run_name "${RUN_NAME}" \
  "$@" > "${LAUNCHER_LOG}" 2>&1 & disown

echo "Started v20_mamba_first_weak."
echo "launcher log: ${LAUNCHER_LOG}"