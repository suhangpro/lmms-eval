# Omni3D Evaluation on ORD Cluster

Scripts for running Omni3D 3D object detection evaluation on the NVIDIA ORD Slurm cluster.

## Prerequisites

1. Access to the ORD cluster (cs-oci-ord-login-01, etc.)
2. Huggingface account (`HF_TOKEN`)

## Quick Start

### 1. First-time setup

SSH to the login node and run the setup script:

```bash
ssh cs-oci-ord-login-01

# Download and run setup (or copy from local)
curl -sL https://raw.githubusercontent.com/suhangpro/lmms-eval/omni3d_bench/lmms_eval/tasks/omni3d/slurm/setup_cluster.sh | bash
```

Or if you have the script locally:

```bash
bash /path/to/setup_cluster.sh
```

### 2. Configure data paths

Edit `config.sh` to set your Omni3D data locations:

```bash
cd /lustre/fsw/portfolios/nvr/users/$USER/lmms-eval/lmms_eval/tasks/omni3d/slurm
vi config.sh
```

Update these variables:
- `OMNI3D_IMAGE_ROOT`: Path to Omni3D images on lustre
- `OMNI3D_JSON_ROOT`: Path to Omni3D JSON files on lustre

### 3. Test interactively (recommended first)

Before submitting batch jobs, test interactively:

```bash
# Start interactive shell in container
./interactive.sh

# Follow inline instructions to test
```

### 4. Run batch evaluation

**Option A: Native Transformers (1 GPU)**
```bash
# Quick test
sbatch eval_omni3d.slurm omni3d_test_toy

# Single dataset
sbatch eval_omni3d.slurm omni3d_arkitscenes_test

# All 6 datasets
sbatch eval_omni3d.slurm omni3d_test
```

**Option B: vLLM (8 GPUs, Recommended for speed)**
```bash
# Quick test
sbatch eval_omni3d_vllm.slurm omni3d_test_toy

# Single dataset
sbatch eval_omni3d_vllm.slurm omni3d_arkitscenes_test

# All 6 datasets
sbatch eval_omni3d_vllm.slurm omni3d_test
```

### 5. Monitor and visualize

```bash
# Check job status
squeue -u $USER

# View logs
tail -f outputs/slurm_logs/omni3d_eval_<JOB_ID>.out

# After completion, visualize results
sbatch visualize_omni3d.slurm outputs/omni3d_arkitscenes_test 10
```

## Files

| File | Description |
|------|-------------|
| `config.sh` | Configuration (container, paths, model settings) |
| `setup_cluster.sh` | First-time cluster setup script |
| `interactive.sh` | Interactive testing before batch jobs |
| `eval_omni3d.slurm` | Slurm batch script for native transformers (1 GPU) |
| `eval_omni3d_vllm.slurm` | Slurm batch script for vLLM (8 GPUs, faster) |
| `visualize_omni3d.slurm` | Slurm batch script for visualization |

## Evaluation Options

### Option 1: Native Transformers (Single GPU)

Uses native HuggingFace transformers. Simpler but slower.

```bash
sbatch eval_omni3d.slurm omni3d_test_toy
```

- GPUs: 1
- Batch size: 8
- Speed: ~30 tokens/s (with transformers 5.x)

### Option 2: vLLM (Multi-GPU, Recommended)

Uses vLLM with tensor parallelism. Faster for large evaluations.

```bash
sbatch eval_omni3d_vllm.slurm omni3d_test_toy
```

- GPUs: 8 (tensor_parallel_size=8)
- Batch size: 64
- Speed: Much faster due to optimized kernels

## Configuration Options

Edit `config.sh` to customize:

```bash
# Container
CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3

# Model Settings (used by both scripts)
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct

# Native Transformers (single GPU)
MODEL_TYPE=qwen3_vl
BATCH_SIZE=8

# vLLM Settings (multi-GPU)
VLLM_TENSOR_PARALLEL=8
VLLM_BATCH_SIZE=64
VLLM_GPU_MEMORY_UTIL=0.85

# Data Paths (UPDATE THESE)
OMNI3D_IMAGE_ROOT=/path/to/omni3d_images
OMNI3D_JSON_ROOT=/path/to/Omni3D_json
```

To change Slurm account/partition, edit the `#SBATCH` directives directly in the `.slurm` files.

## Available Tasks

| Task | Description | Samples |
|------|-------------|---------|
| `omni3d_test_toy` | Quick test (ARKitScenes + Hypersim) | 20 |
| `omni3d_arkitscenes_test` | ARKitScenes indoor | ~2k |
| `omni3d_hypersim_test` | Hypersim synthetic | ~5k |
| `omni3d_kitti_test` | KITTI driving | ~7k |
| `omni3d_nuscenes_test` | nuScenes driving | ~6k |
| `omni3d_objectron_test` | Objectron objects | ~15k |
| `omni3d_sunrgbd_test` | SUNRGBD indoor | ~5k |
| `omni3d_test` | All 6 datasets | ~40k |

## Output Structure

```
outputs/
├── slurm_logs/
│   ├── omni3d_eval_12345.out
│   └── omni3d_eval_12345.err
├── omni3d_arkitscenes_test/
│   └── Qwen__Qwen3-VL-8B-Instruct/
│       ├── results.json
│       └── *_samples_*.jsonl
└── omni3d_arkitscenes_test_vis/
    ├── 0_sample.png
    └── ...
```

## Troubleshooting

### Docker credentials error
```bash
# Test container access
srun --partition interactive --gpus 1 --pty \
    --container-image nvcr.io/nvidia/pytorch:25.01-py3 nvidia-smi
```

If this fails, set up Docker/enroot credentials per your cluster documentation.

### Home quota exceeded
```bash
# Check quota
lfs quota -h /home

# Move cache to lustre
mkdir -p /lustre/fsw/portfolios/nvr/users/$USER/cache
rm -rf ~/.cache
ln -s /lustre/fsw/portfolios/nvr/users/$USER/cache ~/.cache
```

### Out of memory

**For native transformers:**
- Reduce `BATCH_SIZE` in config.sh

**For vLLM:**
- Increase `VLLM_GPU_MEMORY_UTIL` (e.g., 0.9)
- Reduce `VLLM_BATCH_SIZE` (e.g., 32)
- Reduce `max_num_seqs` (e.g., 8)
- Reduce `max_model_len` (e.g., 4096)
