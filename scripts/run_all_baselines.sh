#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/configs/baseline_common.yaml"
NUM_GPUS=${1:-8}

echo "========================================"
echo "  Baseline Experiments Runner"
echo "  Config: $CONFIG"
echo "  GPUs: $NUM_GPUS"
echo "========================================"

for METHOD in gap abmil transmil dsmil; do
    LOG_DIR="$PROJECT_DIR/volumetric_logs_baseline_${METHOD}"
    CKPT_DIR="$PROJECT_DIR/volumetric_checkpoints_baseline_${METHOD}"
    LOG_FILE="$LOG_DIR/train.log"

    mkdir -p "$LOG_DIR" "$CKPT_DIR"

    echo ""
    echo "========================================"
    echo "  Starting: $METHOD"
    echo "  Log: $LOG_FILE"
    echo "========================================"

    torchrun --nproc_per_node=$NUM_GPUS --master_port=$((29500 + RANDOM % 100)) \
        "$SCRIPT_DIR/train_baseline_ddp.py" \
        --config "$CONFIG" \
        --aggregation_method "$METHOD" \
        --mixed_precision \
        2>&1 | tee "$LOG_FILE"

    echo "  $METHOD done."
    echo "========================================"
done

echo ""
echo "All baselines complete."
