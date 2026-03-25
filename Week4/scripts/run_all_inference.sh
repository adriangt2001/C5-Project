#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --mem 8192
#SBATCH --gres gpu:1
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"
mkdir -p logs results/task1

CONFIGS=(
    "configs/task1/vit-gpt2-pretrained.yaml"
    "configs/task1/blip-base-pretrained.yaml"
    "configs/task1/blip-large-pretrained.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Running inference with config: $CONFIG"
    echo "========================================"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$PYTHON_BIN" -m src.main inference --config "$CONFIG"
    echo "Done: $CONFIG"
done

echo "All inferences finished!"