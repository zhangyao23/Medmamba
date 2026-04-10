#!/bin/bash
set -e

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MEM_THRESHOLD="${1:-1000}"
POLL_INTERVAL=60
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$PROJECT_DIR/configs/fullsup_seg.yaml"
LOG_DIR="$PROJECT_DIR/volumetric_logs_fullsup_seg"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NPROC=${#GPU_LIST[@]}

check_gpus_free() {
    for gpu_id in "${GPU_LIST[@]}"; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
        if [ -z "$used" ] || [ "$used" -ge "$MEM_THRESHOLD" ]; then
            return 1
        fi
    done
    return 0
}

echo "============================================"
echo "  FullSup Patch Seg - Wait & Train"
echo "============================================"
echo "  GPUs: $GPUS (nproc=$NPROC)"
echo "  Config: $CONFIG"
echo "  Log: $LOG_FILE"
echo "  Mem threshold: ${MEM_THRESHOLD} MiB"
echo "============================================"

echo "[$(date)] Waiting for GPUs [${GPUS}] to be free (< ${MEM_THRESHOLD} MiB each)..."
while ! check_gpus_free; do
    for gpu_id in "${GPU_LIST[@]}"; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
        echo -n "  GPU${gpu_id}: ${used}MiB"
    done
    echo " -- still busy, next check in ${POLL_INTERVAL}s"
    sleep "$POLL_INTERVAL"
done

echo "[$(date)] All GPUs free. Starting training in 10s..."
sleep 10

echo "[$(date)] Launching torchrun on GPUs=$GPUS"

cd "$PROJECT_DIR"
CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    scripts/train_fullsup_seg_ddp.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE"

echo "[$(date)] Training finished. Log saved to $LOG_FILE"
