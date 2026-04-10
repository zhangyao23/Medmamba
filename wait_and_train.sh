#!/bin/bash
set -euo pipefail
trap '' HUP

GPU_IDS="${GPU_IDS:-1,2,3,4,5,6,7}"
REQUIRED_FREE_MB="${REQUIRED_FREE_MB:-20000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
RUN_NAME="${RUN_NAME:-volumetric_v2_waited}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LAUNCHER_LOG="${ARTIFACT_ROOT}/${RUN_NAME}/logs/launcher.log"
RESUME_CKPT="${RESUME_CKPT:-}"

mkdir -p "$(dirname "${LAUNCHER_LOG}")"

echo "Waiting for GPUs [${GPU_IDS}] to have >= ${REQUIRED_FREE_MB} MiB free each..."
echo "Checking every ${CHECK_INTERVAL} seconds..."

while true; do
    all_free=true
    for gid in $(echo "$GPU_IDS" | tr ',' ' '); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gid" 2>/dev/null | tr -d ' ')
        if [ -z "$free" ] || [ "$free" -lt "$REQUIRED_FREE_MB" ]; then
            all_free=false
            break
        fi
    done

    if $all_free; then
        echo "$(date '+%F %T') All GPUs free. Launching training..."
        break
    fi

    echo "$(date '+%F %T') GPU $gid only has ${free:-?} MiB free. Waiting..."
    sleep "$CHECK_INTERVAL"
done

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node="${NUM_GPUS:-7}"
  --master_port="${MASTER_PORT:-29611}"
  scripts/train_volumetric_ddp.py
  --config configs/volumetric_v2_config.yaml
  --batch_size "${BATCH_SIZE:-4}"
  --mini_batch_size "${MINI_BATCH:-8}"
  --max_patches "${MAX_PATCHES:-64}"
  --mixed_precision
  --output_root "${ARTIFACT_ROOT}"
  --run_name "${RUN_NAME}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  cmd+=(--resume "${RESUME_CKPT}")
fi

nohup "${cmd[@]}" >> "${LAUNCHER_LOG}" 2>&1 &
TRAIN_PID=$!
echo "$(date '+%F %T') Training launched (PID: ${TRAIN_PID})"
echo "Log: tail -f ${LAUNCHER_LOG}"