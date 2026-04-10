#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/configs/baseline_common.yaml"
TRAIN_SCRIPT="$SCRIPT_DIR/train_baseline_ddp.py"

echo "========================================"
echo "  Baseline Experiments - Parallel Runner"
echo "  Phase 1: GAP (GPU 0-3) + ABMIL (GPU 4-7)"
echo "  Phase 2: TransMIL (GPU 0-3) + DSMIL (GPU 4-7)"
echo "========================================"

run_baseline() {
    local METHOD=$1
    local GPUS=$2
    local MASTER_PORT=$3
    local NUM_GPU=$4

    local LOG_DIR="$PROJECT_DIR/volumetric_logs_baseline_${METHOD}"
    local CKPT_DIR="$PROJECT_DIR/volumetric_checkpoints_baseline_${METHOD}"
    local LOG_FILE="$LOG_DIR/train.log"

    mkdir -p "$LOG_DIR" "$CKPT_DIR"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $METHOD on GPU $GPUS (port $MASTER_PORT)"

    CUDA_VISIBLE_DEVICES=$GPUS torchrun \
        --nproc_per_node=$NUM_GPU \
        --master_port=$MASTER_PORT \
        "$TRAIN_SCRIPT" \
        --config "$CONFIG" \
        --aggregation_method "$METHOD" \
        --mixed_precision \
        2>&1 | tee "$LOG_FILE"
}

echo ""
echo "=== Phase 1: GAP + ABMIL ==="
run_baseline gap "0,1,2,3" 29501 4 &
PID_GAP=$!
run_baseline abmil "4,5,6,7" 29502 4 &
PID_ABMIL=$!

echo "Waiting for Phase 1 (PIDs: $PID_GAP, $PID_ABMIL)..."
wait $PID_GAP
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GAP finished."
wait $PID_ABMIL
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ABMIL finished."

echo ""
echo "=== Phase 2: TransMIL + DSMIL ==="
run_baseline transmil "0,1,2,3" 29503 4 &
PID_TRANSMIL=$!
run_baseline dsmil "4,5,6,7" 29504 4 &
PID_DSMIL=$!

echo "Waiting for Phase 2 (PIDs: $PID_TRANSMIL, $PID_DSMIL)..."
wait $PID_TRANSMIL
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TransMIL finished."
wait $PID_DSMIL
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DSMIL finished."

echo ""
echo "========================================"
echo "  All baselines complete!"
echo "========================================"
