#!/usr/bin/env python3
"""Visualize predictions vs ground truth 3D bounding boxes for Omni3D."""

import json
import os
import re
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

# Import paths from config
from lmms_eval.tasks.omni3d.config import IMAGE_ROOT, JSON_ROOT


def get_cuboid_verts(center, dimensions, rotation_deg=None):
    """Get 3D cuboid vertices from center and dimensions.
    
    IMPORTANT: This follows VLMEvalKit's convention exactly.
    
    Args:
        center: [x, y, z] center position
        dimensions: [x_size, y_size, z_size] dimensions
        rotation_deg: [roll, pitch, yaw] in degrees (optional) - note: from model output
    
    Returns:
        vertices: 8x3 array of corner points
    """
    x, y, z = center
    x_size, y_size, z_size = dimensions
    
    # Half dimensions
    hx, hy, hz = x_size / 2, y_size / 2, z_size / 2
    
    # 8 corners in local coordinates - VLMEvalKit ordering
    corners = np.array([
        [hx, hy, hz], [hx, hy, -hz], [hx, -hy, hz], [hx, -hy, -hz],
        [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, hz], [-hx, -hy, -hz],
    ])
    
    # Apply rotation if provided
    # Model outputs: [roll, pitch, yaw] but VLMEvalKit interprets bbox_3d[6:9] as [pitch, yaw, roll]
    # and then applies rotation in order: pitch (X), yaw (Y), roll (Z)
    if rotation_deg is not None:
        # Model output is [roll, pitch, yaw], reorder to [pitch, yaw, roll] for VLMEvalKit convention
        roll_deg, pitch_deg, yaw_deg = rotation_deg
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)
        roll = np.radians(roll_deg)
        
        # Apply rotations in VLMEvalKit order: pitch (X), yaw (Y), roll (Z)
        rotated_corners = []
        for corner in corners:
            x0, y0, z0 = corner
            # Pitch rotation (around X axis)
            x1 = x0
            y1 = y0 * np.cos(pitch) - z0 * np.sin(pitch)
            z1 = y0 * np.sin(pitch) + z0 * np.cos(pitch)
            # Yaw rotation (around Y axis)
            x2 = x1 * np.cos(yaw) + z1 * np.sin(yaw)
            y2 = y1
            z2 = -x1 * np.sin(yaw) + z1 * np.cos(yaw)
            # Roll rotation (around Z axis)
            x3 = x2 * np.cos(roll) - y2 * np.sin(roll)
            y3 = x2 * np.sin(roll) + y2 * np.cos(roll)
            z3 = z2
            rotated_corners.append([x3, y3, z3])
        corners = np.array(rotated_corners)
    
    # Translate to center
    vertices = corners + np.array([x, y, z])
    return vertices


def convert_3dbbox_vlmevalkit(bbox_3d, K, scale_rotation=True, debug=False):
    """EXACT copy of VLMEvalKit's convert_3dbbox + draw_3dbboxes logic.
    
    This matches visualize_omni3d_results.py EXACTLY, including:
    - Line 147: multiply rotations by 180
    - Line 103: interpret positions 6,7,8 as pitch, yaw, roll (note: prompt says roll, pitch, yaw!)
    
    Args:
        bbox_3d: [x, y, z, x_size, y_size, z_size, r1, r2, r3] from model
        K: Camera intrinsics matrix (3x3)
        scale_rotation: If True, multiply rotations by 180 (VLMEvalKit default)
        debug: Print debug info
        
    Returns:
        List of 2D corner coordinates (may have <8 if some behind camera)
    """
    import math
    
    bbox_3d = list(bbox_3d)
    
    # VLMEvalKit line 147: scale rotations by 180
    if scale_rotation:
        bbox_3d[-3:] = [_x * 180 for _x in bbox_3d[-3:]]
    
    # VLMEvalKit line 103: unpack (note the order!)
    # They read [6],[7],[8] as pitch, yaw, roll
    # But prompt asks model to output roll, pitch, yaw
    # This is a bug/inconsistency in VLMEvalKit, but we copy it exactly
    x, y, z, x_size, y_size, z_size, pitch, yaw, roll = bbox_3d[:9]
    
    if debug:
        print(f"  VLMEvalKit interp: center=({x:.2f}, {y:.2f}, {z:.2f}), "
              f"size=({x_size:.2f}, {y_size:.2f}, {z_size:.2f}), "
              f"pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f} deg")
    
    hx, hy, hz = x_size / 2, y_size / 2, z_size / 2
    local_corners = [
        [hx, hy, hz], [hx, hy, -hz], [hx, -hy, hz], [hx, -hy, -hz],
        [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, hz], [-hx, -hy, -hz],
    ]
    
    # VLMEvalKit's rotate_xyz function (lines 110-121)
    def rotate_xyz(_point, _pitch, _yaw, _roll):
        x0, y0, z0 = _point
        x1 = x0
        y1 = y0 * math.cos(_pitch) - z0 * math.sin(_pitch)
        z1 = y0 * math.sin(_pitch) + z0 * math.cos(_pitch)
        x2 = x1 * math.cos(_yaw) + z1 * math.sin(_yaw)
        y2 = y1
        z2 = -x1 * math.sin(_yaw) + z1 * math.cos(_yaw)
        x3 = x2 * math.cos(_roll) - y2 * math.sin(_roll)
        y3 = x2 * math.sin(_roll) + y2 * math.cos(_roll)
        z3 = z2
        return [x3, y3, z3]
    
    # VLMEvalKit lines 123-131: project to 2D
    K = np.array(K)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    img_corners = []
    for corner in local_corners:
        rotated = rotate_xyz(corner, np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll))
        X, Y, Z = rotated[0] + x, rotated[1] + y, rotated[2] + z
        if Z > 0:
            x_2d = fx * (X / Z) + cx
            y_2d = fy * (Y / Z) + cy
            img_corners.append([x_2d, y_2d])
    
    return img_corners


def get_cuboid_verts_vlmevalkit(bbox_3d, scale_rotation=True, debug=False):
    """Convert bbox_3d to 3D vertices using VLMEvalKit's method.
    
    Args:
        bbox_3d: [x, y, z, x_size, y_size, z_size, r1, r2, r3] from model
        scale_rotation: If True, multiply rotations by 180
        debug: Print debug info
        
    Returns:
        vertices: 8x3 array of corner points
    """
    import math
    
    bbox_3d = list(bbox_3d)
    
    # VLMEvalKit line 147: scale rotations by 180
    if scale_rotation:
        bbox_3d[-3:] = [_x * 180 for _x in bbox_3d[-3:]]
    
    # VLMEvalKit reads positions 6,7,8 as pitch, yaw, roll
    x, y, z, x_size, y_size, z_size, pitch, yaw, roll = bbox_3d[:9]
    
    if debug:
        print(f"  bbox_3d: center=({x:.2f}, {y:.2f}, {z:.2f}), "
              f"size=({x_size:.2f}, {y_size:.2f}, {z_size:.2f}), "
              f"pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f} deg")
    
    hx, hy, hz = x_size / 2, y_size / 2, z_size / 2
    local_corners = [
        [hx, hy, hz], [hx, hy, -hz], [hx, -hy, hz], [hx, -hy, -hz],
        [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, hz], [-hx, -hy, -hz],
    ]
    
    def rotate_xyz(_point, _pitch, _yaw, _roll):
        x0, y0, z0 = _point
        x1 = x0
        y1 = y0 * math.cos(_pitch) - z0 * math.sin(_pitch)
        z1 = y0 * math.sin(_pitch) + z0 * math.cos(_pitch)
        x2 = x1 * math.cos(_yaw) + z1 * math.sin(_yaw)
        y2 = y1
        z2 = -x1 * math.sin(_yaw) + z1 * math.cos(_yaw)
        x3 = x2 * math.cos(_roll) - y2 * math.sin(_roll)
        y3 = x2 * math.sin(_roll) + y2 * math.cos(_roll)
        z3 = z2
        return [x3, y3, z3]
    
    corners_3d = []
    for corner in local_corners:
        rotated = rotate_xyz(corner, np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll))
        corners_3d.append([rotated[0] + x, rotated[1] + y, rotated[2] + z])
    
    return np.array(corners_3d)


def project_3d_to_2d(points_3d, K):
    """Project 3D points to 2D using camera intrinsics K."""
    K = np.array(K)
    points_2d = []
    
    for pt in points_3d:
        x, y, z = pt
        if z <= 0:
            z = 0.01
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        points_2d.append([u, v])
    
    return np.array(points_2d)


def draw_cuboid(draw, vertices_2d, color, line_width=2, label=None):
    """Draw 3D cuboid edges on image.
    
    Uses VLMEvalKit's edge connectivity for their corner ordering:
    [hx,hy,hz], [hx,hy,-hz], [hx,-hy,hz], [hx,-hy,-hz],
    [-hx,hy,hz], [-hx,hy,-hz], [-hx,-hy,hz], [-hx,-hy,-hz]
    """
    # VLMEvalKit edge connectivity
    edges = [
        [0, 1], [2, 3], [4, 5], [6, 7],  # edges along Z
        [0, 2], [1, 3], [4, 6], [5, 7],  # edges along Y  
        [0, 4], [1, 5], [2, 6], [3, 7],  # edges along X
    ]
    
    for i, j in edges:
        try:
            pt1 = tuple(vertices_2d[i].astype(int))
            pt2 = tuple(vertices_2d[j].astype(int))
            draw.line([pt1, pt2], fill=color, width=line_width)
        except (IndexError, ValueError):
            continue
    
    # Draw corners
    for pt in vertices_2d:
        try:
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x-4, y-4, x+4, y+4], fill=color)
        except (ValueError, TypeError):
            continue
    
    # Draw label at top corner
    if label and len(vertices_2d) > 0:
        try:
            top_pt = vertices_2d[np.argmin(vertices_2d[:, 1])]
            draw.text((int(top_pt[0]) + 5, int(top_pt[1]) - 15), label, fill=color)
        except (ValueError, IndexError):
            pass


def draw_cuboid_gt(draw, vertices_2d, color, line_width=2, label=None):
    """Draw 3D cuboid edges for GT annotations (bbox3D_cam format).
    
    GT bbox3D_cam uses standard Omni3D/cubercnn corner ordering which is different
    from VLMEvalKit's prediction corner ordering.
    
    Standard ordering (from cubercnn):
    [-l/2, -h/2, -w/2], [l/2, -h/2, -w/2], [l/2, h/2, -w/2], [-l/2, h/2, -w/2],
    [-l/2, -h/2, w/2], [l/2, -h/2, w/2], [l/2, h/2, w/2], [-l/2, h/2, w/2]
    """
    # Standard Omni3D edge connectivity (same as visualize_gt.py)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # back face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]
    
    for i, j in edges:
        try:
            pt1 = tuple(vertices_2d[i].astype(int))
            pt2 = tuple(vertices_2d[j].astype(int))
            draw.line([pt1, pt2], fill=color, width=line_width)
        except (IndexError, ValueError):
            continue
    
    # Draw corners
    for pt in vertices_2d:
        try:
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x-4, y-4, x+4, y+4], fill=color)
        except (ValueError, TypeError):
            continue
    
    # Draw label at top corner
    if label and len(vertices_2d) > 0:
        try:
            top_pt = vertices_2d[np.argmin(vertices_2d[:, 1])]
            draw.text((int(top_pt[0]) + 5, int(top_pt[1]) - 15), label, fill=color)
        except (ValueError, IndexError):
            pass


def parse_prediction(response_text):
    """Parse prediction from model response.
    
    Returns list of dicts with bbox_3d and label.
    """
    # Try to extract JSON from response
    text = response_text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = re.sub(r'^```\s*', '', text)
    
    try:
        predictions = json.loads(text)
        if isinstance(predictions, list):
            return predictions
    except json.JSONDecodeError:
        pass
    
    return []


def visualize_predictions(samples_file, output_dir, num_samples=10, dataset_name="Objectron", 
                          random_sample=False, seed=42, scale_rotation=True, quiet=False):
    """Visualize predictions from a samples JSONL file.
    
    Args:
        samples_file: Path to samples JSONL file
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
        dataset_name: Dataset name for loading GT
        random_sample: If True, randomly select samples instead of first N
        seed: Random seed for reproducibility
        scale_rotation: If True, multiply rotations by 180 (VLMEvalKit does this)
        quiet: If True, suppress debug output (raw responses, bbox values, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load samples
    print(f"Loading samples from {samples_file}...")
    with open(samples_file) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples")
    
    # Random sampling if requested
    if random_sample and len(samples) > num_samples:
        random.seed(seed)
        samples = random.sample(samples, num_samples)
        print(f"Randomly selected {num_samples} samples (seed={seed})")
    
    # Load GT data
    json_path = os.path.join(JSON_ROOT, f"{dataset_name}_test.json")
    print(f"Loading GT from {json_path}...")
    with open(json_path) as f:
        gt_data = json.load(f)
    
    imgid2info = {img["id"]: img for img in gt_data["images"]}
    imgid2anns = {}
    for ann in gt_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in imgid2anns:
            imgid2anns[img_id] = []
        imgid2anns[img_id].append(ann)
    
    # Process samples
    count = 0
    stats = {"total": 0, "has_pred": 0, "has_valid": 0}
    
    # Limit samples if not already randomly sampled
    samples_to_process = samples if random_sample else samples[:num_samples]
    
    for sample in samples_to_process:
        stats["total"] += 1
        
        # Get image info
        image_id = sample["omni3d_valid_rate"]["image_id"]
        category = sample["omni3d_valid_rate"]["category"]
        response = sample["filtered_resps"][0]
        
        if image_id not in imgid2info:
            print(f"Warning: Image {image_id} not found in GT data")
            continue
        
        img_info = imgid2info[image_id]
        image_path = os.path.join(IMAGE_ROOT, img_info["file_path"])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        K = img_info["K"]
        
        # Get GT annotations for this image and category
        gt_anns = [a for a in imgid2anns.get(image_id, []) 
                   if a["category_name"] == category]
        
        # Draw GT bboxes (green)
        for j, ann in enumerate(gt_anns):
            vertices_3d = np.array(ann["bbox3D_cam"])
            
            # Debug: print GT info
            if not quiet:
                center = ann.get("center_cam", [0,0,0])
                dims = ann.get("dimensions", [0,0,0])
                print(f"  GT {j+1}: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                      f"dims=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f})")
                print(f"    bbox3D_cam corners[0]: {vertices_3d[0]}")
            
            vertices_2d = project_3d_to_2d(vertices_3d, K)
            # Use GT-specific edge connectivity (different from VLMEvalKit prediction corners)
            draw_cuboid_gt(draw, vertices_2d, color=(0, 255, 0), line_width=3, label=f"GT{j+1}")
        
        # Parse and draw predictions (red/blue)
        predictions = parse_prediction(response)
        
        if predictions:
            stats["has_pred"] += 1
            if not quiet:
                print(f"\n  Image {image_id}, category={category}:")
                print(f"  Raw response: {response[:200]}..." if len(response) > 200 else f"  Raw response: {response}")
        
        for i, pred in enumerate(predictions):
            bbox_3d = pred.get("bbox_3d", [])
            if len(bbox_3d) >= 6:
                stats["has_valid"] += 1
                
                # Ensure we have all 9 values (pad with zeros for rotation if needed)
                if len(bbox_3d) < 9:
                    bbox_3d = list(bbox_3d) + [0] * (9 - len(bbox_3d))
                
                if not quiet:
                    print(f"  Pred {i+1} raw: {bbox_3d}")
                
                # Use EXACT VLMEvalKit method (includes projection)
                vertices_2d = convert_3dbbox_vlmevalkit(bbox_3d, K, scale_rotation=scale_rotation, debug=not quiet)
                
                if len(vertices_2d) >= 8:
                    vertices_2d = np.array(vertices_2d)
                    draw_cuboid(draw, vertices_2d, color=(255, 0, 0), line_width=2, label=f"Pred{i+1}")
                elif not quiet:
                    print(f"    Warning: only {len(vertices_2d)} corners visible (some behind camera)")
        
        # Add title
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        title = f"{category} | GT: {len(gt_anns)} | Pred: {len(predictions)}"
        draw.rectangle([0, 0, image.width, 35], fill=(0, 0, 0))
        draw.text((10, 5), title, fill=(255, 255, 255), font=font)
        
        # Add legend
        legend_y = 45
        draw.rectangle([5, legend_y, 80, legend_y + 25], outline=(0, 255, 0), width=2)
        draw.text((85, legend_y + 3), "Ground Truth", fill=(0, 255, 0))
        draw.rectangle([5, legend_y + 30, 80, legend_y + 55], outline=(255, 0, 0), width=2)
        draw.text((85, legend_y + 33), "Prediction", fill=(255, 0, 0))
        
        # Save
        out_path = os.path.join(output_dir, f"pred_{count+1}_{category}_{image_id}.jpg")
        image.save(out_path)
        if not quiet:
            print(f"Saved: {out_path}")
        
        count += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"Total samples visualized: {stats['total']}")
    print(f"Samples with predictions: {stats['has_pred']} ({100*stats['has_pred']/max(1,stats['total']):.1f}%)")
    print(f"Valid predictions: {stats['has_valid']}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Omni3D predictions vs GT")
    parser.add_argument("--samples_file", required=True, help="Path to samples JSONL file")
    parser.add_argument("--output_dir", default="./outputs/omni3d_pred_vis", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--dataset", default="Objectron", help="Dataset name")
    parser.add_argument("--random", action="store_true", help="Randomly select samples instead of first N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_scale_rotation", action="store_true", 
                       help="Don't multiply rotations by 180 (default: scale by 180 like VLMEvalKit)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress debug output (raw responses, bbox values, etc.)")
    
    args = parser.parse_args()
    scale_rotation = not args.no_scale_rotation
    visualize_predictions(args.samples_file, args.output_dir, args.num_samples, args.dataset, 
                         args.random, args.seed, scale_rotation, args.quiet)
