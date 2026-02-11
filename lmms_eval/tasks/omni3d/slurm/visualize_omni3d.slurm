#!/bin/bash
#SBATCH --job-name=omni3d_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --account=nvr_lpr_nvgptvision
#SBATCH --partition=grizzly,polar,polar3,polar4,batch_singlenode
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ============================================================================
# Omni3D Visualization Job Script for Cluster
# 
# Usage:
#   sbatch visualize_omni3d.slurm <results_dir> [num_samples]
#
# Examples:
#   sbatch visualize_omni3d.slurm outputs/omni3d_arkitscenes_test 10
#   sbatch visualize_omni3d.slurm outputs/omni3d_test_toy 5 --random
# ============================================================================

# Get arguments
RESULTS_DIR="${1:-outputs/omni3d_test_toy}"
NUM_SAMPLES="${2:-10}"
EXTRA_ARGS="${@:3}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
source "${SCRIPT_DIR}/config.sh"

echo "=============================================="
echo "Omni3D Visualization Job"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "=============================================="

# Find the samples file
SAMPLES_FILE=$(find "${PROJECT_ROOT}/${RESULTS_DIR}" -name "*.jsonl" | head -1)
if [ -z "${SAMPLES_FILE}" ]; then
    echo "ERROR: No samples file found in ${RESULTS_DIR}"
    exit 1
fi
echo "Samples file: ${SAMPLES_FILE}"

# Determine output directory
VIS_OUTPUT_DIR="${PROJECT_ROOT}/${RESULTS_DIR}_vis"
mkdir -p "${VIS_OUTPUT_DIR}"

# Move log files to log directory
mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "${LOG_DIR}/" 2>/dev/null || true
mv "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "${LOG_DIR}/" 2>/dev/null || true

# Run visualization in container
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${PROJECT_ROOT}:/workspace,/lustre:/lustre" \
     --container-workdir="/workspace" \
     --no-container-entrypoint \
     bash -c "
        set -e
        echo 'Installing dependencies...'
        pip install -q -e ".[all]"
        
        echo 'Setting environment variables...'
        export OMNI3D_IMAGE_ROOT='${OMNI3D_IMAGE_ROOT}'
        export OMNI3D_JSON_ROOT='${OMNI3D_JSON_ROOT}'
        
        echo 'Running visualization...'
        python lmms_eval/tasks/omni3d/visualize_predictions.py \
            --samples_file '${SAMPLES_FILE}' \
            --num_samples ${NUM_SAMPLES} \
            --output_dir '${VIS_OUTPUT_DIR}' \
            ${EXTRA_ARGS}
        
        echo 'Visualization complete!'
     "

echo "=============================================="
echo "Job completed!"
echo "Visualizations saved to: ${VIS_OUTPUT_DIR}"
echo "=============================================="
