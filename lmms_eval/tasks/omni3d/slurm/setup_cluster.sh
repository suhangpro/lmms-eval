#!/bin/bash
# ============================================================================
# Omni3D Cluster Setup Script
# 
# Run this script on the ORD cluster to set up the lmms-eval environment
# with the Omni3D benchmark.
#
# Usage:
#   bash setup_cluster.sh
# ============================================================================

set -e

echo "=============================================="
echo "Omni3D Cluster Setup"
echo "=============================================="

# Configuration
LUSTRE_WORKSPACE_DIR="/lustre/fsw/portfolios/nvr/users/${USER}"
CACHE_DIR="${LUSTRE_WORKSPACE_DIR}/cache"
REPO_URL="git@github.com:suhangpro/lmms-eval.git"
BRANCH="omni3d_bench"

# Enter directory
cd "${LUSTRE_WORKSPACE_DIR}"

# Clone or update repository
if [ -d "lmms-eval" ]; then
    echo "Repository exists. Updating..."
    cd lmms-eval
    git fetch origin
    git checkout ${BRANCH}
    git pull origin ${BRANCH}
else
    echo "Cloning repository..."
    git clone -b ${BRANCH} ${REPO_URL}
    cd lmms-eval
fi

echo "Repository ready at: $(pwd)"

# Set up cache symlink to avoid filling home quota
echo "Setting up cache directory..."
mkdir -p "${CACHE_DIR}/huggingface"
mkdir -p "${CACHE_DIR}/torch"

# Create symlinks if they don't exist
if [ ! -L "${HOME}/.cache" ]; then
    if [ -d "${HOME}/.cache" ]; then
        echo "Moving existing cache to lustre..."
        mv "${HOME}/.cache"/* "${CACHE_DIR}/" 2>/dev/null || true
        rm -rf "${HOME}/.cache"
    fi
    ln -sf "${CACHE_DIR}" "${HOME}/.cache"
    echo "Symlinked ~/.cache -> ${CACHE_DIR}"
fi

# Create output directories
mkdir -p outputs/slurm_logs

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Edit config.sh to set your Omni3D data paths:"
echo "   vi ${LUSTRE_WORKSPACE_DIR}/lmms-eval/lmms_eval/tasks/omni3d/slurm/config.sh"
echo ""
echo "2. Run a test evaluation:"
echo "   cd ${LUSTRE_WORKSPACE_DIR}/lmms-eval/lmms_eval/tasks/omni3d/slurm"
echo "   sbatch eval_omni3d.slurm omni3d_test_toy"
echo ""
echo "3. Monitor your job:"
echo "   squeue -u \$USER"
echo "=============================================="
