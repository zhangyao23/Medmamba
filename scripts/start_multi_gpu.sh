#!/bin/bash
set -o pipefail

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
MAX_PATCHES="${MAX_PATCHES:-64}"
MASTER_PORT="${MASTER_PORT:-29500}"
ENABLE_AMP="${ENABLE_AMP:-1}"
RESUME_CKPT="${RESUME_CKPT:-}"

AMP_FLAG=""
if [ "$ENABLE_AMP" = "1" ]; then
    AMP_FLAG="--mixed_precision"
fi

RESUME_FLAG=""
if [ -n "$RESUME_CKPT" ]; then
    RESUME_FLAG="--resume ${RESUME_CKPT}"
fi

cd /mnt/nas/share/home/liuke/prjs/uter/model_with_mamba

echo "=========================================="
echo "Multi-GPU Training (7 GPUs)"
echo "=========================================="
echo ""
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | grep -E "^[1-7],"
echo ""
echo "Configuration:"
echo "  - World size: ${NPROC_PER_NODE} GPUs"
echo "  - Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "  - Total batch size: $((NPROC_PER_NODE * BATCH_SIZE_PER_GPU))"
echo "  - Feature extractor mini-batch size: ${MINI_BATCH_SIZE}"
echo "  - Max patches per volume: ${MAX_PATCHES}"
echo "  - Mixed precision: ${ENABLE_AMP}"
echo "  - Resume checkpoint: ${RESUME_CKPT:-none}"
echo "  - Backend: NCCL"
echo ""
echo "=========================================="
echo ""

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    scripts/train_volumetric_ddp.py \
    --config configs/volumetric_config.yaml \
    --batch_size "${BATCH_SIZE_PER_GPU}" \
    --mini_batch_size "${MINI_BATCH_SIZE}" \
    --max_patches "${MAX_PATCHES}" \
    ${RESUME_FLAG} \
    ${AMP_FLAG} 2>&1
