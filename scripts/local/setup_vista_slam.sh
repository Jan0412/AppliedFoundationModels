#!/bin/bash
set -e

# Resolve repository root from the script's current location
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

echo "=== Starting Setup of ViSTA-SLAM ==="

echo "=== Initializing Git Submodules ==="
git submodule update --init --recursive

echo "=== Synchronizing Virtual Environment ==="
uv sync

echo "=== Downloading Pretrained Weights ==="
uv run hf download zhangganlin/vista_slam frontend_sta_weights.pth ORBvoc.txt --local-dir external/vista-slam/pretrains

echo "=== Compiling DBoW3 C++ Bindings ==="
uv pip install --no-build-isolation ./external/vista-slam/DBoW3Py

echo "=== Compiling CuRoPE CUDA Kernels ==="
cd external/vista-slam/vista_slam/sta_model/pos_embed/curope

# Force LD_LIBRARY_PATH inline to link both cuDNN and NCCL
LD_LIBRARY_PATH="$ROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$ROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH" "$ROOT_DIR/.venv/bin/python" setup.py build_ext --inplace

cd "$ROOT_DIR"

echo "=== Completed Setup of ViSTA-SLAM ==="