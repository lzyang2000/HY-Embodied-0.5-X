#!/bin/bash
# =============================================================
# setup_env.sh — One-click conda environment setup for HY-Embodied-0.5-X
#
# Usage:
#   bash setup_env.sh
#
# After setup:
#   conda activate hy_embodied_x
#   python inference.py --model ckpts/HY-Embodied-0.5-X --image ./assets/demo.jpg --prompt "Describe this image" --no-thinking
# =============================================================

set -e

# ---- Config ----
ENV_NAME="hy_embodied_x"
PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Colors ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- Step 0: Check conda ----
info "Checking conda..."
if ! command -v conda &>/dev/null; then
    error "conda not found. Please install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
fi
info "conda version: $(conda --version)"

# ---- Step 1: Create conda env ----
if conda env list | grep -qw "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists, skipping creation"
else
    info "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# ---- Step 2: Activate env ----
info "Activating environment: ${ENV_NAME}..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
info "Python: $(python --version) @ $(which python)"

# ---- Step 3: Upgrade pip ----
info "Upgrading pip..."
pip install --upgrade pip

# ---- Step 4: Install PyTorch (must be installed BEFORE flash_attn) ----
info "Installing PyTorch + torchvision + triton..."
pip install torch==2.10.0 torchvision==0.25.0 triton==3.6.0 --index-url https://download.pytorch.org/whl/cu126

info "Verifying PyTorch..."
python -c "import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || error "PyTorch installation failed"

# ---- Step 5: Install flash_attn build dependencies ----
info "Installing flash_attn build dependencies (ninja, psutil, setuptools, wheel)..."
pip install ninja==1.13.0 psutil==7.2.1 "setuptools>=70.1" wheel

# ---- Step 6: Install flash_attn (builds from source, takes 10-20 min) ----
info "Installing flash_attn==2.8.3 (compiling from source, this may take 10-20 minutes)..."
pip install flash_attn==2.8.3 --no-build-isolation

# ---- Step 7: Install transformers (specific commit with HY-Embodied model registration) ----
info "Installing transformers (specific commit with HY-Embodied support)..."
pip install "git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a"

# ---- Step 8: Install remaining dependencies ----
info "Installing remaining dependencies..."
pip install \
    accelerate==1.12.0 \
    deepspeed==0.18.4 \
    huggingface-hub==0.36.0 \
    safetensors==0.7.0 \
    tokenizers==0.22.2 \
    hf-xet==1.2.0 \
    timm==1.0.21 \
    liger-kernel==0.7.0 \
    Pillow \
    opencv-python==4.13.0.90 \
    numpy==2.4.1 \
    einops==0.8.1 \
    tqdm==4.67.1 \
    packaging==26.0 \
    regex==2026.1.15 \
    requests==2.32.5 \
    filelock==3.20.3 \
    fsspec==2026.1.0 \
    typing_extensions==4.15.0 \
    PyYAML==6.0.3 \
    Jinja2==3.1.6 \
    MarkupSafe==3.0.3 \
    sympy==1.14.0 \
    mpmath==1.3.0 \
    networkx==3.6.1 \
    certifi==2026.1.4 \
    charset-normalizer==3.4.4 \
    idna==3.11 \
    urllib3==2.6.3 \
    pydantic==2.12.5 \
    pydantic_core==2.41.5 \
    py-cpuinfo==9.0.0 \
    hjson==3.1.0 \
    msgpack==1.1.2 \
    tensorboard==2.20.0 \
    tensorboard-data-server==0.7.2 \
    protobuf \
    grpcio==1.76.0 \
    Markdown==3.10.1 \
    Werkzeug==3.1.5 \
    absl-py==2.3.1 \
    diffusers==0.34.0 \
    fastapi==0.115.12 \
    "uvicorn[standard]==0.34.3"

# ---- Step 9: Verify key packages ----
info "Verifying key packages..."
python -c "
import torch, transformers, flash_attn, timm
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  flash_attn:   {flash_attn.__version__}')
print(f'  timm:         {timm.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
" || error "Package verification failed"

# ---- Done ----
echo ""
info "=========================================="
info "  Setup complete!"
info "=========================================="
echo ""
echo "  Quick start:"
echo ""
echo "    conda activate ${ENV_NAME}"
echo "    cd ${SCRIPT_DIR}"
echo "    python inference.py --model ckpts/HY-Embodied-0.5-X --image ./assets/demo.jpg --prompt \"Describe this image\" --no-thinking"
echo ""
