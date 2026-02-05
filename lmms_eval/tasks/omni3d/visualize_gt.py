#!/usr/bin/env python3
"""Visualize ground truth 3D bounding boxes for Omni3D debugging."""

import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

# Import paths from config
from lmms_eval.tasks.omni3d.config import IMAGE_ROOT, JSON_ROOT


def get_cuboid_verts_faces(bbox3d):
    """Get 3D cuboid vertices and faces from bbox3d.
    
    bbox3d: [x, y, z, w, h, l, roll, pitch, yaw] (center, dimensions, rotation in degrees)
    Returns: vertices (8x3), faces (6x4 vertex indices)
    """
    x, y, z, w, h, l, roll, pitch, yaw = bbox3d
    
    # Convert rotation angles from degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Create rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    
    # Half dimensions
    hw, hh, hl = w/2, h/2, l/2
    
    # 8 corners of the cuboid (in local coordinates)
    corners = np.array([
        [-hw, -hh, -hl],
        [+hw, -hh, -hl],
        [+hw, +hh, -hl],
        [-hw, +hh, -hl],
        [-hw, -hh, +hl],
        [+hw, -hh, +hl],
        [+hw, +hh, +hl],
        [-hw, +hh, +hl],
    ])
    
    # Rotate and translate
    vertices = (R @ corners.T).T + np.array([x, y, z])
    
    # Faces (front, back, left, right, top, bottom)
    faces = np.array([
        [0, 1, 2, 3],  # front
        [4, 5, 6, 7],  # back
        [0, 4, 7, 3],  # left
        [1, 5, 6, 2],  # right
        [3, 2, 6, 7],  # top
        [0, 1, 5, 4],  # bottom
    ])
    
    return vertices, faces


def project_3d_to_2d(points_3d, K):
    """Project 3D points to 2D using camera intrinsics K."""
    # K is 3x3 intrinsic matrix
    K = np.array(K)
    points_2d = []
    
    for pt in points_3d:
        x, y, z = pt
        if z <= 0:
            z = 0.01  # Avoid division by zero
        
        # Project: u = fx * x/z + cx, v = fy * y/z + cy
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
        points_2d.append([u, v])
    
    return np.array(points_2d)


def draw_cuboid_on_image(image, vertices_2d, color=(0, 255, 0), line_width=2):
    """Draw 3D cuboid edges on image."""
    draw = ImageDraw.Draw(image)
    
    # Edges of the cuboid (connect vertices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # back face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]
    
    for i, j in edges:
        pt1 = tuple(vertices_2d[i].astype(int))
        pt2 = tuple(vertices_2d[j].astype(int))
        draw.line([pt1, pt2], fill=color, width=line_width)
    
    # Draw corners as dots
    for pt in vertices_2d:
        x, y = int(pt[0]), int(pt[1])
        draw.ellipse([x-3, y-3, x+3, y+3], fill=color)
    
    return image


def visualize_sample(dataset_name, category, num_samples=5, output_dir="./outputs/omni3d_vis", random_sample=False, seed=42):
    """Visualize GT 3D bboxes for samples of a category.
    
    Args:
        dataset_name: Dataset name (e.g., "Objectron", "ARKitScenes")
        category: Category to visualize (e.g., "cup", "chair")
        num_samples: Number of samples to visualize
        output_dir: Output directory for visualizations
        random_sample: If True, randomly select samples instead of first N
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    json_path = os.path.join(JSON_ROOT, f"{dataset_name}_test.json")
    print(f"Loading {json_path}...")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Build image lookup
    imgid2info = {img["id"]: img for img in data["images"]}
    
    # Filter annotations by category
    cat_anns = [ann for ann in data["annotations"] if ann["category_name"] == category]
    print(f"Found {len(cat_anns)} annotations for '{category}'")
    
    if len(cat_anns) == 0:
        print(f"No annotations found for category '{category}'")
        return
    
    # Get unique images with this category
    unique_images = list(set(ann["image_id"] for ann in cat_anns))
    print(f"Found {len(unique_images)} unique images with '{category}'")
    
    # Random sampling if requested
    if random_sample and len(unique_images) > num_samples:
        random.seed(seed)
        selected_images = random.sample(unique_images, num_samples)
        print(f"Randomly selected {num_samples} images (seed={seed})")
    else:
        selected_images = unique_images[:num_samples]
    
    # Visualize samples
    count = 0
    seen_images = set()
    
    for ann in cat_anns:
        if count >= num_samples:
            break
        
        image_id = ann["image_id"]
        
        # Skip if not in selected images (for random mode)
        if random_sample and image_id not in selected_images:
            continue
        
        if image_id in seen_images:
            continue
        seen_images.add(image_id)
        
        img_info = imgid2info[image_id]
        image_path = os.path.join(IMAGE_ROOT, img_info["file_path"])
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Get camera intrinsics
        K = img_info.get("K", [[1, 0, image.width/2], [0, 1, image.height/2], [0, 0, 1]])
        
        # Get all annotations for this image with this category
        img_cat_anns = [a for a in cat_anns if a["image_id"] == image_id]
        
        print(f"\nImage {count+1}: {img_info['file_path']}")
        print(f"  Resolution: {image.width}x{image.height}")
        print(f"  Annotations: {len(img_cat_anns)}")
        
        # Draw all bboxes
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, a in enumerate(img_cat_anns):
            # Omni3D uses different format:
            # - center_cam: [x, y, z] center position
            # - dimensions: [w, h, l] dimensions  
            # - bbox3D_cam: 8 corner points already computed
            center = a["center_cam"]
            dims = a["dimensions"]
            bbox3d_corners = np.array(a["bbox3D_cam"])  # 8x3 corners
            
            print(f"  Bbox3D #{i+1}: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                  f"dims=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f})")
            
            # Use pre-computed 3D corners
            vertices_3d = bbox3d_corners
            
            # Project to 2D
            vertices_2d = project_3d_to_2d(vertices_3d, K)
            
            # Draw on image
            color = colors[i % len(colors)]
            image = draw_cuboid_on_image(image, vertices_2d, color=color, line_width=3)
        
        # Add label
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), f"{category} (GT)", fill=(255, 255, 255), font=font)
        
        # Save
        output_path = os.path.join(output_dir, f"{dataset_name}_{category}_{count+1}.jpg")
        image.save(output_path)
        print(f"  Saved: {output_path}")
        
        count += 1
    
    print(f"\nVisualization complete! {count} images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Omni3D GT 3D bboxes")
    parser.add_argument("--dataset", default="Objectron", help="Dataset name")
    parser.add_argument("--category", default="cup", help="Category to visualize")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--output_dir", default="./outputs/omni3d_vis", help="Output directory")
    parser.add_argument("--random", action="store_true", help="Randomly select samples instead of first N")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    visualize_sample(args.dataset, args.category, args.num_samples, args.output_dir, args.random, args.seed)
