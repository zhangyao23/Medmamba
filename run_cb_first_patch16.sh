#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
RUN_NAME="${RUN_NAME:-v20_cb_first_patch16}"
PYTHON_BIN="${PYTHON_BIN:-/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE:-2}" \
    --master_port="${MASTER_PORT:-29621}" \
    scripts/train_cb_first_patch16_ddp.py \
    --config configs/v20_cb_first_patch16.yaml \
    --output_root "${ARTIFACT_ROOT}" \
    --run_name "${RUN_NAME}" \
    "$@"