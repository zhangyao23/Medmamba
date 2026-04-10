#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/mamba_artifacts}"
PYTHON_BIN="${PYTHON_BIN:-/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/bin/python}"
NUM_GPUS="${NUM_GPUS:-5}"
WAIT_PATTERN="${WAIT_PATTERN:-train_volumetric_ddp.*v20_mamba_first_weak}"

cd "$REPO_ROOT"

export LD_LIBRARY_PATH=/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

find_free_gpus() {
    local needed=$1
    local free_gpus=""
    local count=0
    for gpu_id in 0 1 2 3 4 5 6 7; do
        local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d ' ')
        if [ -n "$mem_used" ] && [ "$mem_used" -lt 1000 ]; then
            if [ -z "$free_gpus" ]; then
                free_gpus="$gpu_id"
            else
                free_gpus="$free_gpus,$gpu_id"
            fi
            count=$((count + 1))
            if [ "$count" -ge "$needed" ]; then
                break
            fi
        fi
    done
    if [ "$count" -lt "$needed" ]; then
        echo ""
        return 1
    fi
    echo "$free_gpus"
    return 0
}

run_experiment() {
    local config=$1
    local run_name=$2
    local master_port=$3
    local log_dir="${ARTIFACT_ROOT}/${run_name}/logs"
    local logfile="${log_dir}/launcher.log"

    mkdir -p "$log_dir"

    echo "=============================================="
    echo "Waiting for $NUM_GPUS free GPUs..."
    echo "=============================================="
    while true; do
        GPU_IDS=$(find_free_gpus "$NUM_GPUS")
        if [ -n "$GPU_IDS" ]; then
            break
        fi
        echo "$(date): Not enough free GPUs, retrying in 60s..."
        sleep 60
    done

    echo "=============================================="
    echo "Starting experiment: $config"
    echo "  Run:  $run_name"
    echo "  GPUs: $GPU_IDS"
    echo "  Log:  $logfile"
    echo "  Time: $(date)"
    echo "=============================================="

    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$master_port" \
        scripts/train_volumetric_ddp.py \
        --config "$config" \
        --output_root "$ARTIFACT_ROOT" \
        --run_name "$run_name" \
        > "$logfile" 2>&1

    echo "Experiment finished: $config (exit code: $?)"
    echo "  Time: $(date)"
}

echo "=============================================="
echo "Waiting for current training to finish..."
echo "=============================================="
while true; do
    if ! pgrep -f "$WAIT_PATTERN" > /dev/null 2>&1; then
        echo "$(date): Current Mamba-first training finished."
        break
    fi
    echo "$(date): Training still running, checking again in 120s..."
    sleep 120
done

sleep 30

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> Experiment 1/2: Codebook-first + Frozen Partition"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
run_experiment \
    "configs/v20_cb_first_frozen_partition.yaml" \
    "cb_first_frozen_partition" \
    29613

sleep 30

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> Experiment 2/2: Mamba-first + Frozen Partition"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
run_experiment \
    "configs/v20_mamba_first_frozen_partition.yaml" \
    "mamba_first_frozen_partition" \
    29614

echo ""
echo "=============================================="
echo "All frozen partition experiments completed!"
echo "  Time: $(date)"
echo "=============================================="
