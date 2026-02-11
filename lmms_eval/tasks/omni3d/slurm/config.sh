#!/bin/bash
# ============================================================================
# Omni3D Evaluation Configuration for ORD Cluster
# ============================================================================

# Container Configuration
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"

# Project Paths (on lustre)
export PROJECT_ROOT="/lustre/fsw/portfolios/nvr/users/${USER}/lmms-eval"
export OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
export LOG_DIR="${OUTPUT_ROOT}/slurm_logs"

# Omni3D Data Paths - UPDATE THESE FOR YOUR SETUP
export OMNI3D_IMAGE_ROOT="${OMNI3D_IMAGE_ROOT:-/lustre/fsw/portfolios/nvr/users/sifeil/datasets/omni3d_bench/omni3d_images}"
export OMNI3D_JSON_ROOT="${OMNI3D_JSON_ROOT:-/lustre/fsw/portfolios/nvr/users/sifeil/datasets/omni3d_bench/Omni3D_json}"

# Model Configuration
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"

# Native Transformers (single GPU)
export MODEL_TYPE="${MODEL_TYPE:-qwen3_vl}"
export BATCH_SIZE="${BATCH_SIZE:-8}"

# vLLM Configuration (multi-GPU)
export VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-8}"
export VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-64}"
export VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.85}"

# HuggingFace Configuration
export HF_HOME="/lustre/fsw/portfolios/nvr/users/${USER}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${HF_HOME}"
