#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
RUN_NAME="${RUN_NAME:-eval_c1_epoch78}"
PYTHON_BIN="${PYTHON_BIN:-/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE:-1}" \
    --master_port="${MASTER_PORT:-29622}" \
    scripts/train_cb_first_mamba_pretrain_ddp.py \
    --config configs/eval_c1_epoch78.yaml \
    --output_root "${ARTIFACT_ROOT}" \
    --run_name "${RUN_NAME}" \
    "$@"