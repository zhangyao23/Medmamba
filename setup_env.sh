#!/bin/bash
set -e

ENV_NAME="uter_mamba"
PYTHON_VER="3.11"

echo "=== Step 1: Creating conda environment ==="
conda create -n $ENV_NAME python=$PYTHON_VER -y

ENVPIP="$(conda info --base)/envs/$ENV_NAME/bin/pip"
ENVPYTHON="$(conda info --base)/envs/$ENV_NAME/bin/python"

echo "=== Step 2: Installing PyTorch (CUDA 12.8) ==="
$ENVPIP install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== Step 3: Installing core dependencies ==="
$ENVPIP install \
  nibabel==5.3.3 \
  monai==1.5.2 \
  einops==0.8.2 \
  numpy==1.26.4 \
  pandas==2.1.4 \
  scikit-learn==1.3.2 \
  matplotlib==3.8.1 \
  seaborn==0.13.2 \
  plotly==6.5.2 \
  pyvista==0.46.5 \
  tqdm==4.65.0 \
  PyYAML==6.0.1 \
  tensorboard==2.20.0 \
  timm==1.0.24 \
  scipy

echo "=== Step 4: Installing causal-conv1d ==="
$ENVPIP install causal-conv1d==1.6.0

echo "=== Step 5: Installing mamba-ssm ==="
$ENVPIP install mamba-ssm==2.3.0

echo "=== Step 6: Verifying installation ==="
$ENVPYTHON -c "
import torch; print(f'torch {torch.__version__} CUDA={torch.cuda.is_available()}')
import mamba_ssm; print(f'mamba_ssm {mamba_ssm.__version__}')
import causal_conv1d; print(f'causal_conv1d {causal_conv1d.__version__}')
import monai; print(f'monai {monai.__version__}')
import nibabel; print(f'nibabel {nibabel.__version__}')
import einops; print(f'einops {einops.__version__}')
import timm; print(f'timm {timm.__version__}')
print('=== All packages verified OK ===')
"

echo ""
echo "=== Setup complete! ==="
echo "To activate:  conda activate $ENV_NAME"
echo "To train:     see run_training.sh"
