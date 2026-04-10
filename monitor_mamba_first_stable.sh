#!/bin/bash
set -euo pipefail

# 自动监控 Mamba-first stable 训练
# 每30分钟检查一次，如果崩溃则报告错误

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
RUN_NAME="${RUN_NAME:-mamba_first_stable}"
PROCESS_PATTERN="${PROCESS_PATTERN:-train_mamba_first_ddp}"
LOG_DIR="${ARTIFACT_ROOT}/${RUN_NAME}/logs"
LOG="${LOG_DIR}/train.log"
MONITOR_LOG="${LOG_DIR}/monitor.log"

mkdir -p "$LOG_DIR"
echo "$(date): Monitor started" >> "$MONITOR_LOG"

while true; do
    sleep 1800  # 30 minutes

    if [ ! -f "$LOG" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): waiting for log file at $LOG" >> "$MONITOR_LOG"
        continue
    fi

    RUNNING=$(pgrep -fc "$PROCESS_PATTERN" || true)
    LAST_LOSS=$(grep -oP 'loss=[\d.]+' "$LOG" | tail -1 || true)
    LAST_EPOCH=$(grep -oP 'Epoch \d+' "$LOG" | tail -1 || true)
    HAS_NAN=$(grep "Train Loss: nan" "$LOG" | tail -1 || true)
    HAS_ERROR=$(tail -20 "$LOG" | grep -iE "Error|Traceback|RuntimeError|NCCL" | head -1 || true)

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$RUNNING" -gt 0 ] && [ -z "$HAS_NAN" ] && [ -z "$HAS_ERROR" ]; then
        echo "$TIMESTAMP: OK | $LAST_EPOCH | $LAST_LOSS | procs=$RUNNING" >> "$MONITOR_LOG"
    else
        echo "$TIMESTAMP: PROBLEM DETECTED" >> "$MONITOR_LOG"
        echo "  Running processes: $RUNNING" >> "$MONITOR_LOG"
        echo "  Last epoch: $LAST_EPOCH" >> "$MONITOR_LOG"
        echo "  Last loss: $LAST_LOSS" >> "$MONITOR_LOG"
        echo "  NaN detected: $HAS_NAN" >> "$MONITOR_LOG"
        echo "  Error: $HAS_ERROR" >> "$MONITOR_LOG"
        echo "  --- Last 10 lines of log ---" >> "$MONITOR_LOG"
        tail -10 "$LOG" >> "$MONITOR_LOG"
        echo "  --- END ---" >> "$MONITOR_LOG"
        echo "$TIMESTAMP: STOPPED - needs manual intervention" >> "$MONITOR_LOG"
        exit 1
    fi
done
