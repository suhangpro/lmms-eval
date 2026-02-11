#!/bin/bash
#SBATCH --job-name=omni3d_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:59:59
#SBATCH --account=nvr_lpr_nvgptvision
#SBATCH --partition=grizzly,polar,polar3,polar4,batch_singlenode
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================================
# Omni3D Evaluation Job Script for Cluster
# 
# Usage:
#   sbatch eval_omni3d.slurm <task_name>
#
# Examples:
#   sbatch eval_omni3d.slurm omni3d_test_toy          # Quick test
#   sbatch eval_omni3d.slurm omni3d_arkitscenes_test  # Single dataset
#   sbatch eval_omni3d.slurm omni3d_test              # All datasets
# ============================================================================

# Get task name from argument
TASK_NAME="${1:-omni3d_test_toy}"

# Configuration - hardcoded paths for Slurm compatibility
# (BASH_SOURCE doesn't work reliably in Slurm spool)
CONFIG_DIR="/lustre/fsw/portfolios/nvr/users/${USER}/lmms-eval/lmms_eval/tasks/omni3d/slurm"

# Load configuration if exists, otherwise use defaults
if [ -f "${CONFIG_DIR}/config.sh" ]; then
    source "${CONFIG_DIR}/config.sh"
else
    echo "Warning: config.sh not found, using defaults"
    export CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"
    export PROJECT_ROOT="/lustre/fsw/portfolios/nvr/users/${USER}/lmms-eval"
    export OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
    export LOG_DIR="${OUTPUT_ROOT}/slurm_logs"
    export OMNI3D_IMAGE_ROOT="${OMNI3D_IMAGE_ROOT:-/lustre/fsw/portfolios/nvr/users/sifeil/datasets/omni3d_bench/omni3d_images}"
    export OMNI3D_JSON_ROOT="${OMNI3D_JSON_ROOT:-/lustre/fsw/portfolios/nvr/users/sifeil/datasets/omni3d_bench/Omni3D_json}"
    export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
    export MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
    export BATCH_SIZE="${BATCH_SIZE:-8}"
    export HF_HOME="/lustre/fsw/portfolios/nvr/users/${USER}/cache/huggingface"
    export TRANSFORMERS_CACHE="${HF_HOME}/hub"
fi

echo "=============================================="
echo "Omni3D Evaluation Job"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Task: ${TASK_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Model Type: ${MODEL_TYPE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Container: ${CONTAINER_IMAGE}"
echo "=============================================="

# Create output directory
OUTPUT_DIR="${OUTPUT_ROOT}/${TASK_NAME}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Move log files to log directory
mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${LOG_DIR}/" 2>/dev/null || true
mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${LOG_DIR}/" 2>/dev/null || true

# Run evaluation in container
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${PROJECT_ROOT}:/workspace,/lustre:/lustre" \
     --container-workdir="/workspace" \
     --no-container-entrypoint \
     bash -c "
        set -e
        echo 'Installing dependencies...'
        pip install -q -e '.[all]'
        
        echo 'Setting environment variables...'
        export OMNI3D_IMAGE_ROOT='${OMNI3D_IMAGE_ROOT}'
        export OMNI3D_JSON_ROOT='${OMNI3D_JSON_ROOT}'
        export HF_HOME='${HF_HOME}'
        export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
        export HF_TOKEN='${HF_TOKEN}'
        
        echo 'Starting evaluation...'
        python -m lmms_eval \
            --model ${MODEL_TYPE} \
            --model_args pretrained=${MODEL_NAME} \
            --tasks ${TASK_NAME} \
            --batch_size ${BATCH_SIZE} \
            --log_samples \
            --output_path ${OUTPUT_DIR}
        
        echo 'Evaluation complete!'
     "

echo "=============================================="
echo "Job completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
