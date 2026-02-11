# Omni3D: 3D Object Detection Benchmark

## Overview

Omni3D is a comprehensive benchmark for evaluating 3D object detection capabilities of vision-language models from single RGB images. This implementation supports all 6 Omni3D datasets with official AP3D/AR3D metrics.

## Task Format

**Input:** Single RGB image  
**Output:** JSON list of 3D bounding boxes

```json
[
  {
    "bbox_3d": [x, y, z, width, height, length, roll, pitch, yaw],
    "label": "chair"
  }
]
```

Where:
- `x, y, z`: Object center in camera coordinates (meters), z = depth
- `width, height, length`: Object dimensions in meters
- `roll, pitch, yaw`: Rotation angles

> **Note on dimension format:** `[width, height, length]` will be interpreted as `[size_x, size_y, size_z]`.
>
> **Note on angle format:** Despite the prompt saying `[roll, pitch, yaw]`, the output is interpreted as `[pitch, yaw, roll]` (positions 6, 7, 8 in bbox_3d). The angles are within a normalized [-1, 1] range, and will be scaled by 180 internally. 

## Datasets

| Dataset | Task Name | Domain | Example Categories |
|---------|-----------|--------|-------------------|
| ARKitScenes | `omni3d_arkitscenes_test` | Indoor AR scenes | cabinet, chair, table, sofa |
| Hypersim | `omni3d_hypersim_test` | Synthetic indoor | chair, table, sofa, bed |
| KITTI | `omni3d_kitti_test` | Autonomous driving | car, pedestrian, cyclist |
| nuScenes | `omni3d_nuscenes_test` | Autonomous driving | car, truck, bus, pedestrian |
| Objectron | `omni3d_objectron_test` | Object-centric mobile | chair, cup, laptop, camera |
| SUNRGBD | `omni3d_sunrgbd_test` | Indoor RGB-D | bed, chair, desk, table |

**Total:** 97 object categories across all datasets

### Task Groups

- `omni3d_test`: All 6 datasets

## Setup

### Data Directory Structure

```
/path/to/omni3d_images/          # Image files
├── ARKitScenes/...
├── hypersim/...
├── KITTI_object/...
├── nuScenes/...
├── objectron/...
└── SUNRGBD/...

/path/to/Omni3D_json/            # JSON annotations
├── ARKitScenes_test.json
├── Hypersim_test.json
├── KITTI_test.json
├── nuScenes_test.json
├── Objectron_test.json
└── SUNRGBD_test.json
```

Image paths are specified in the JSON files and resolved relative to `OMNI3D_IMAGE_ROOT`.

### Configuration

Set environment variables to point to your data:

```bash
export OMNI3D_IMAGE_ROOT="/path/to/omni3d_images"
export OMNI3D_JSON_ROOT="/path/to/Omni3D_json"
```

Or edit `config.py` directly to change the default paths.

## Usage

### Basic Evaluation

Run evaluation on a single dataset:

```bash
python -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct \
    --tasks omni3d_arkitscenes_test \
    --batch_size 1 \
    --log_samples \
    --output_path ./outputs/omni3d_arkitscenes
```

Run on all Omni3D datasets:

```bash
python -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct \
    --tasks omni3d_test \
    --batch_size 1 \
    --log_samples \
    --output_path ./outputs/omni3d_all
```

### Visualization

Visualize ground truth:

```bash
python lmms_eval/tasks/omni3d/visualize_gt.py \
    --dataset ARKitScenes \
    --num_samples 5 \
    --output_dir ./outputs/gt_vis
```

Visualize predictions vs ground truth:

```bash
python lmms_eval/tasks/omni3d/visualize_predictions.py \
    --samples_file outputs/omni3d_arkitscenes/*/samples_omni3d_arkitscenes_test.jsonl \
    --num_samples 10 \
    --output_dir ./outputs/pred_vis
```

Use `--random` for random sampling instead of sequential.

### Cluster Deployment (Slurm)

For running evaluations on Slurm clusters with multi-GPU support and vLLM, see the [Omni3D Evaluation on ORD Cluster](slurm/README.md).

## Evaluation Metrics

### Basic Metrics
- **Valid Prediction Rate**: Percentage of predictions with valid bbox format
- **Recall Proxy**: Percentage of samples with at least one valid prediction

### Official AP3D/AR3D Metrics
- **AP3D@15/25/50**: Average Precision at 3D IoU thresholds 0.15, 0.25, 0.50
- **AR3D@1/10/100**: Average Recall at various max detection limits

The AP3D metrics use precise 3D IoU computation via convex hull intersection (no external dependencies required).

## Prompt Format

The prompt sent to the model follows format:

```
Locate the chair in the provided image and output their positions and dimensions using 3D bounding boxes. The results must be in the JSON format: `[{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"chair"}]`. If no chair is visible, output `[]`.
```

## Coordinate System

- **Camera Coordinates**: Right-handed, +Z forward (depth), +X right, +Y down
- **Bounding Box**: Center + dimensions + rotation (Euler angles)
- **Units**: Meters for positions/dimensions, degrees (with normalization) for angles (see "Note on angle format" above)

## Filtering Criteria

Following Omni3D official protocol, annotations are filtered based on:
- Visibility ≥ 0.33
- Truncation ≤ 0.33
- Object height: 0.0625 × image_height to 1.5 × image_height
- Valid 3D annotation (valid3D = True)
- Not behind camera
- Positive dimensions
- Positive depth

## Performance Notes

- **Batch Size**: Recommend batch_size=1 for VLM models
- **Speed**: Expect ~1-5 seconds per image on a single GPU depending on model and hardware
- **Memory**: Models like Qwen3-VL-8B need ~16GB GPU memory

### vLLM vs Native Transformers Inference

**Important:** Installing vLLM may slow down native transformers-based inference, and therefore for a single-gpu setup vLLM is perhaps not a good choice.

**Why this happens:** vLLM 0.11.0 has a dependency constraint requiring `transformers<5.0`, which downgrades to 4.57.6. The older transformers version lacks performance optimizations for vision-language models.

#### Option 1: Use vLLM for inference (requires tuning)

```bash
uv run python -m lmms_eval \
    --model vllm \
    --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,max_num_seqs=8 \
    --tasks omni3d_arkitscenes_test \
    --batch_size 1 \
    --log_samples \
    --output_path ./outputs/omni3d_arkitscenes_test_vllm
```

**Note:** vLLM on 24GB GPUs (e.g., 3090) requires careful memory tuning. Key parameters:
- `pretrained=Qwen/Qwen3-VL-4B-Instruct`: Use 4B model
- `gpu_memory_utilization=0.9`: Fraction of VRAM for vLLM
- `max_model_len=4096`: Limit context length to reduce KV cache memory
- `max_num_seqs=8`: Limit concurrent sequences

#### Option 2: Remove vLLM for faster native inference (recommended for local)

For local development on consumer GPUs, native transformers inference is often simpler and faster:

```bash
# Remove vLLM and restore newer transformers
uv remove vllm
uv add transformers --upgrade-package transformers
uv add torch --upgrade-package torch

# Verify versions (should be transformers>=5.1.0, torch>=2.10.0)
uv pip list | grep -E "(transformers|torch)"

# Run with native inference
uv run python -m lmms_eval \
    --model qwen3_vl \
    --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct \
    --tasks omni3d_arkitscenes_test \
    --batch_size 1 \
    --log_samples \
    --output_path ./outputs/omni3d_arkitscenes_test
```

#### Recommendation

| Environment | Recommended Setup |
|-------------|-------------------|
| Local dev (single GPU, 24GB) | Native transformers 5.x (no vLLM) |
| Multi-GPU cluster | vLLM with tensor_parallel_size>1 |
| Production serving | vLLM with optimized settings |

## References

- [Omni3D GitHub](https://github.com/facebookresearch/omni3d)
- [Omni3D Paper](https://arxiv.org/abs/2207.10660)
- [Internal VLMEvalKit Implementation](https://gitlab-master.nvidia.com/dir/forks/vlmevalkit)

