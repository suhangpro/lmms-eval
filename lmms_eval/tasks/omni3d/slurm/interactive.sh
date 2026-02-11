#!/bin/bash
# ============================================================================
# Interactive Slurm Session for Omni3D Evaluation
# 
# Usage:
#   ./interactive.sh
#
# Once inside the container, you can run:
#   python -m lmms_eval --model qwen3_vl --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct \
#       --tasks omni3d_test_toy --batch_size 8 --log_samples --output_path outputs/test
#   or
#   python -m lmms_eval --model vllm --model_args model=Qwen/Qwen3-VL-8B-Instruct,tensor_parallel_size=8,data_parallel_size=1,gpu_memory_utilization=0.85 \
#       --tasks omni3d_test_toy --batch_size 64 --log_samples --output_path outputs/test
# ============================================================================

set -e

# Get script directory and load config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Slurm allocation settings
ACCOUNT="nvr_lpr_nvgptvision"
PARTITION="grizzly,polar,polar3,polar4,batch_singlenode"
TIME="03:59:59"
GPUS=8

echo "=============================================="
echo "Starting Interactive Slurm Session"
echo "=============================================="
echo "Account: ${ACCOUNT}"
echo "Partition: ${PARTITION}"
echo "Time: ${TIME}"
echo "GPUs: ${GPUS}"
echo "Container: ${CONTAINER_IMAGE}"
echo "=============================================="

# Start interactive shell
echo "Starting interactive shell in container..."
echo ""
echo "IMPORTANT: Always run commands from /workspace (project root)!"
echo ""
echo "Once inside, run these setup commands:"
echo "  cd /workspace                    # IMPORTANT: Must be in project root!"
echo "  uv sync --extra all          # Install dependencies"
echo "  export OMNI3D_IMAGE_ROOT='${OMNI3D_IMAGE_ROOT}'"
echo "  export OMNI3D_JSON_ROOT='${OMNI3D_JSON_ROOT}'"
echo "  export HF_HOME='${HF_HOME}'"
echo ""
echo "Then run evaluation:"
echo "  uv run python -m lmms_eval --model qwen3_vl --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct \\"
echo "      --tasks omni3d_test_toy --batch_size 8 --log_samples --output_path outputs/test"
echo "  or"
echo "  uv run python -m lmms_eval --model vllm --model_args model=Qwen/Qwen3-VL-8B-Instruct,tensor_parallel_size=8,data_parallel_size=1,gpu_memory_utilization=0.85 \\"
echo "      --tasks omni3d_test_toy --batch_size 64 --log_samples --output_path outputs/test_vllm"
echo ""

srun --account="${ACCOUNT}" \
     --partition="${PARTITION}" \
     --time="${TIME}" \
     --gpus-per-node="${GPUS}" \
     --nodes=1 \
     --ntasks-per-node=1 \
     --pty \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${PROJECT_ROOT}:/workspace,/lustre:/lustre" \
     --container-workdir="/workspace" \
     --no-container-entrypoint \
     bash
