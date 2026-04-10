#!/bin/bash
# 自动监控 Mamba-first stable 训练
# 每30分钟检查一次，如果崩溃则报告错误

LOG="/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_final/mamba_first_stable_train.log"
MONITOR_LOG="/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_final/mamba_first_stable_monitor.log"

echo "$(date): Monitor started" >> "$MONITOR_LOG"

while true; do
    sleep 1800  # 30 minutes

    # Check if training process is still running
    RUNNING=$(ps aux | grep "train_mamba_first_ddp" | grep -v grep | wc -l)
    
    # Get latest loss
    LAST_LOSS=$(grep -oP 'loss=[\d.]+' "$LOG" | tail -1)
    LAST_EPOCH=$(grep -oP 'Epoch \d+' "$LOG" | tail -1)
    HAS_NAN=$(grep "Train Loss: nan" "$LOG" | tail -1)
    HAS_ERROR=$(tail -20 "$LOG" | grep -iE "Error|Traceback|RuntimeError|NCCL" | head -1)
    
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
