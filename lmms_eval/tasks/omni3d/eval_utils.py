"""
Evaluation utilities for Omni3D 3D object detection tasks.

This module provides functions for parsing and validating 3D bounding box predictions
from model responses, filtering annotations based on Omni3D criteria, and computing
official AP3D metrics following the Omni3D benchmark.

Reference: https://github.com/facebookresearch/omni3d/blob/main/cubercnn/evaluation/omni3d_evaluation.py
"""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog


def parse_bbox_3d_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse 3D bounding boxes from model response text.
    
    Handles various response formats:
    - JSON code blocks: ```json [...] ```
    - Raw JSON arrays: [...]
    - Truncated responses (extracts complete objects only)
    
    Args:
        text: Model response text containing JSON predictions
    
    Returns:
        List of bbox dicts, each with 'bbox_3d' (9 floats) and 'label' (str)
        Returns empty list if no valid bboxes found.
    
    Example:
        >>> text = '[{"bbox_3d": [0, 0, 5, 1, 1, 2, 0, 0, 0], "label": "chair"}]'
        >>> parse_bbox_3d_from_text(text)
        [{"bbox_3d": [0.0, 0.0, 5.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0], "label": "chair"}]
    """
    if not text or not isinstance(text, str):
        return []
    
    # Extract JSON block from markdown code fences
    if "```json" in text:
        s = text.split("```json", 1)[1]
        s = s.split("```", 1)[0]
    elif "```" in text:
        # Try generic code block
        s = text.split("```", 1)[1]
        s = s.split("```", 1)[0]
    else:
        # Look for JSON array
        start = text.find("[")
        if start == -1:
            return []
        s = text[start:]
    
    s = s.strip()
    
    # Try full parse first
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return _validate_and_normalize_bboxes(data)
        elif isinstance(data, dict):
            return _validate_and_normalize_bboxes([data])
        return []
    except json.JSONDecodeError:
        pass
    
    # Handle truncated JSON: extract only complete objects
    items = []
    depth = 0
    start_idx = None
    
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                obj_str = s[start_idx : i + 1]
                try:
                    obj = json.loads(obj_str)
                    items.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = None
    
    return _validate_and_normalize_bboxes(items)


def _validate_and_normalize_bboxes(bboxes: List[Dict]) -> List[Dict[str, Any]]:
    """Validate and normalize bbox format.
    
    Args:
        bboxes: List of raw bbox dicts
    
    Returns:
        List of validated and normalized bbox dicts
    """
    validated = []
    
    for bbox in bboxes:
        if not isinstance(bbox, dict):
            continue
        
        # Must have bbox_3d and label
        if "bbox_3d" not in bbox or "label" not in bbox:
            continue
        
        bbox_3d = bbox["bbox_3d"]
        label = bbox["label"]
        
        # Validate bbox_3d format
        if not isinstance(bbox_3d, (list, tuple)):
            continue
        if len(bbox_3d) != 9:
            continue
        
        # Convert to floats and check for valid values
        try:
            bbox_3d_float = [float(x) for x in bbox_3d]
            
            # Check for NaN or inf
            if any(np.isnan(x) or np.isinf(x) for x in bbox_3d_float):
                continue
            
            validated.append({
                "bbox_3d": bbox_3d_float,
                "label": str(label).strip()
            })
        except (ValueError, TypeError):
            continue
    
    return validated


def validate_bbox_3d(bbox_3d: List[float]) -> bool:
    """Check if bbox_3d has valid format and reasonable values.
    
    Args:
        bbox_3d: List of 9 floats [x, y, z, w, h, l, roll, pitch, yaw]
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(bbox_3d, (list, tuple)):
        return False
    
    if len(bbox_3d) != 9:
        return False
    
    try:
        bbox_array = np.array(bbox_3d, dtype=np.float32)
        
        # Check for NaN or inf
        if np.any(np.isnan(bbox_array)) or np.any(np.isinf(bbox_array)):
            return False
        
        # Check dimensions are positive (indices 3, 4, 5 are w, h, l)
        if np.any(bbox_array[3:6] <= 0):
            return False
        
        # Check depth (z) is positive
        if bbox_array[2] <= 0:
            return False
        
        # Check reasonable value ranges
        # Position: within [-1000, 1000] meters
        if np.any(np.abs(bbox_array[:3]) > 1000):
            return False
        
        # Dimensions: within [0.01, 100] meters
        if np.any(bbox_array[3:6] < 0.01) or np.any(bbox_array[3:6] > 100):
            return False
        
        # Rotation: within [-360, 360] degrees
        if np.any(np.abs(bbox_array[6:9]) > 360):
            return False
        
        return True
    except (ValueError, TypeError):
        return False


def filter_annotation(
    ann: Dict[str, Any],
    img_info: Dict[str, Any],
    filter_settings: Dict[str, Any]
) -> bool:
    """Check if annotation should be filtered out based on Omni3D criteria.
    
    Args:
        ann: Annotation dict with fields like visibility, truncation, etc.
        img_info: Image info dict with width, height
        filter_settings: Dict with thresholds for filtering
    
    Returns:
        True if annotation should be FILTERED OUT (ignored), False if should be kept
    """
    # Skip if behind camera
    if ann.get("behind_camera", False):
        return True
    
    # Skip if not valid 3D annotation
    if not ann.get("valid3D", False):
        return True
    
    # Check dimensions are positive
    dimensions = ann.get("dimensions", [0, 0, 0])
    if any(d <= 0 for d in dimensions):
        return True
    
    # Check depth
    center_cam = ann.get("center_cam", [0, 0, 0])
    if len(center_cam) < 3:
        return True
    depth = center_cam[2]
    max_depth = filter_settings.get("max_depth", 1e8)
    if depth > max_depth:
        return True
    
    # Check lidar points and segmentation points (if available)
    if ann.get("lidar_pts", 0) == 0:
        return True
    if ann.get("segmentation_pts", 0) == 0:
        return True
    
    # Check depth error (if available)
    depth_error = ann.get("depth_error", -1)
    if depth_error > 0.5:
        return True
    
    # Determine which 2D bbox to use for height checking
    bbox2D = None
    if filter_settings.get("modal_2D_boxes", False):
        if "bbox2D_tight" in ann and ann["bbox2D_tight"][0] != -1:
            bbox2D = _convert_bbox_xyxy_to_xywh(ann["bbox2D_tight"])
    
    if bbox2D is None and filter_settings.get("trunc_2D_boxes", True):
        if "bbox2D_trunc" in ann:
            trunc = ann["bbox2D_trunc"]
            if not all(val == -1 for val in trunc):
                bbox2D = _convert_bbox_xyxy_to_xywh(trunc)
    
    if bbox2D is None and "bbox2D_proj" in ann:
        bbox2D = _convert_bbox_xyxy_to_xywh(ann["bbox2D_proj"])
    
    if bbox2D is None and "bbox" in ann:
        bbox2D = ann["bbox"]  # Assume already in XYWH format
    
    # Check bbox height constraints
    if bbox2D is not None and len(bbox2D) >= 4:
        img_height = img_info.get("height", 1)
        bbox_height = bbox2D[3]
        
        min_height = filter_settings.get("min_height_thres", 0.0625) * img_height
        max_height = filter_settings.get("max_height_thres", 1.5) * img_height
        
        if bbox_height <= min_height or bbox_height >= max_height:
            return True
    
    # Check truncation
    truncation = ann.get("truncation", -1)
    truncation_thres = filter_settings.get("truncation_thres", 0.33333333)
    if truncation >= 0 and truncation >= truncation_thres:
        return True
    
    # Check visibility
    visibility = ann.get("visibility", -1)
    visibility_thres = filter_settings.get("visibility_thres", 0.33333333)
    if visibility >= 0 and visibility <= visibility_thres:
        return True
    
    return False


def _convert_bbox_xyxy_to_xywh(bbox_xyxy: List[float]) -> List[float]:
    """Convert bounding box from XYXY format to XYWH format.
    
    Args:
        bbox_xyxy: [x1, y1, x2, y2]
    
    Returns:
        [x, y, width, height]
    """
    if len(bbox_xyxy) < 4:
        return [0, 0, 0, 0]
    
    x1, y1, x2, y2 = bbox_xyxy[:4]
    return [x1, y1, x2 - x1, y2 - y1]


def convert_bbox_xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    """Convert bounding box from XYWH format to XYXY format.
    
    Args:
        bbox_xywh: [x, y, width, height]
    
    Returns:
        [x1, y1, x2, y2]
    """
    if len(bbox_xywh) < 4:
        return [0, 0, 0, 0]
    
    x, y, w, h = bbox_xywh[:4]
    return [x, y, x + w, y + h]


# ============================================================================
# Official AP3D Metrics - Following Omni3D Benchmark
# Reference: https://github.com/facebookresearch/omni3d
# ============================================================================

# IoU thresholds for 3D evaluation (different from 2D!)
IOU_THRESHOLDS_3D = np.array([0.15, 0.25, 0.50])
# For recall calculation
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
# Max detections per image
MAX_DETS = [1, 10, 100]


def get_iou_method_info() -> str:
    """Get info about the 3D IoU computation method being used.
    
    Returns:
        String describing the IoU method
    """
    return "Precise 3D IoU (convex hull intersection, CPU)"


def bbox_3d_to_corners(bbox_3d: List[float]) -> np.ndarray:
    """Convert 3D bounding box parameterization to 8 corner points.
    
    Args:
        bbox_3d: [x, y, z, w, h, l, roll, pitch, yaw] in camera coordinates
                 where (x,y,z) is center, (w,h,l) are dimensions,
                 (roll, pitch, yaw) are rotations in degrees
    
    Returns:
        corners: (8, 3) array of corner coordinates
        
    Corner ordering (following Omni3D/PyTorch3D convention):
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)
    """
    x, y, z, w, h, l, roll, pitch, yaw = bbox_3d
    
    # Convert rotations from degrees to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # Create rotation matrix from Euler angles (ZYX convention)
    # Rotation order: roll (X), pitch (Y), yaw (Z)
    cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    
    # Combined rotation matrix (ZYX convention)
    R = np.array([
        [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
        [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
        [-sin_p, cos_p * sin_r, cos_p * cos_r]
    ])
    
    # Unit cube corners centered at origin
    # Order follows PyTorch3D convention for iou_box3d
    unit_corners = np.array([
        [-0.5, -0.5, -0.5],  # 0: front-bottom-left
        [0.5, -0.5, -0.5],   # 1: front-bottom-right
        [0.5, 0.5, -0.5],    # 2: front-top-right
        [-0.5, 0.5, -0.5],   # 3: front-top-left
        [-0.5, -0.5, 0.5],   # 4: back-bottom-left
        [0.5, -0.5, 0.5],    # 5: back-bottom-right
        [0.5, 0.5, 0.5],     # 6: back-top-right
        [-0.5, 0.5, 0.5],    # 7: back-top-left
    ])
    
    # Scale by dimensions
    scaled_corners = unit_corners * np.array([w, h, l])
    
    # Rotate corners
    rotated_corners = (R @ scaled_corners.T).T
    
    # Translate to center position
    corners = rotated_corners + np.array([x, y, z])
    
    return corners


def _point_in_box(point: np.ndarray, box_corners: np.ndarray) -> bool:
    """Check if a point is inside a convex box defined by 8 corners.
    
    Uses the half-space method: a point is inside if it's on the correct
    side of all faces of the convex hull.
    
    Args:
        point: (3,) array - point to test
        box_corners: (8, 3) array of box corners
    
    Returns:
        True if point is inside the box
    """
    try:
        hull = ConvexHull(box_corners)
        # Check if point satisfies all half-space constraints
        # Each equation is [a, b, c, d] where ax + by + cz + d <= 0
        for eq in hull.equations:
            if np.dot(eq[:-1], point) + eq[-1] > 1e-8:
                return False
        return True
    except Exception:
        return False


def _get_box_faces(corners: np.ndarray) -> List[np.ndarray]:
    """Get the 6 face planes of a box from its 8 corners.
    
    Each face is defined by its normal vector and a point on the plane.
    
    Args:
        corners: (8, 3) array of corner coordinates
    
    Returns:
        List of (normal, point) tuples for each face
    """
    # Face definitions based on corner indices
    # Assumes standard box corner ordering
    face_indices = [
        [0, 1, 2, 3],  # front
        [4, 5, 6, 7],  # back
        [0, 1, 5, 4],  # bottom
        [2, 3, 7, 6],  # top
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]
    
    faces = []
    for indices in face_indices:
        p0, p1, p2 = corners[indices[0]], corners[indices[1]], corners[indices[2]]
        # Compute normal via cross product
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
            faces.append((normal, p0))
    
    return faces


def _line_plane_intersection(
    p1: np.ndarray, 
    p2: np.ndarray, 
    plane_normal: np.ndarray, 
    plane_point: np.ndarray
) -> Optional[np.ndarray]:
    """Find intersection point of a line segment with a plane.
    
    Args:
        p1, p2: Endpoints of the line segment
        plane_normal: Normal vector of the plane
        plane_point: A point on the plane
    
    Returns:
        Intersection point if exists within segment, else None
    """
    line_dir = p2 - p1
    denom = np.dot(plane_normal, line_dir)
    
    if abs(denom) < 1e-10:
        return None  # Line parallel to plane
    
    t = np.dot(plane_normal, plane_point - p1) / denom
    
    if 0 <= t <= 1:
        return p1 + t * line_dir
    return None


def _get_box_edges(corners: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Get the 12 edges of a box from its 8 corners.
    
    Args:
        corners: (8, 3) array of corner coordinates
    
    Returns:
        List of (start_point, end_point) tuples for each edge
    """
    # Edge definitions based on corner indices
    edge_indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # back face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]
    
    return [(corners[i], corners[j]) for i, j in edge_indices]


def compute_box3d_iou_precise(box1_corners: np.ndarray, box2_corners: np.ndarray) -> float:
    """Compute precise 3D IoU between two oriented bounding boxes.
    
    Uses convex hull intersection to compute the exact intersection volume.
    This method:
    1. Finds vertices of box1 inside box2 and vice versa
    2. Finds edge-face intersection points
    3. Computes convex hull of all intersection points
    4. Calculates the intersection volume
    
    Args:
        box1_corners: (8, 3) array of corner coordinates
        box2_corners: (8, 3) array of corner coordinates
    
    Returns:
        IoU value in [0, 1]
    """
    # Compute box volumes using convex hull
    try:
        vol1 = ConvexHull(box1_corners).volume
        vol2 = ConvexHull(box2_corners).volume
    except Exception:
        # Degenerate boxes
        return 0.0
    
    if vol1 <= 0 or vol2 <= 0:
        return 0.0
    
    # Quick AABB check for early termination
    min1, max1 = box1_corners.min(axis=0), box1_corners.max(axis=0)
    min2, max2 = box2_corners.min(axis=0), box2_corners.max(axis=0)
    
    if np.any(max1 < min2) or np.any(max2 < min1):
        return 0.0  # No intersection possible
    
    # Collect all intersection points
    intersection_points = []
    
    # 1. Add vertices of box1 that are inside box2
    hull2 = ConvexHull(box2_corners)
    for corner in box1_corners:
        inside = True
        for eq in hull2.equations:
            if np.dot(eq[:-1], corner) + eq[-1] > 1e-8:
                inside = False
                break
        if inside:
            intersection_points.append(corner)
    
    # 2. Add vertices of box2 that are inside box1
    hull1 = ConvexHull(box1_corners)
    for corner in box2_corners:
        inside = True
        for eq in hull1.equations:
            if np.dot(eq[:-1], corner) + eq[-1] > 1e-8:
                inside = False
                break
        if inside:
            intersection_points.append(corner)
    
    # 3. Add edge-face intersection points
    edges1 = _get_box_edges(box1_corners)
    edges2 = _get_box_edges(box2_corners)
    
    # Edges of box1 intersecting faces of box2
    for p1, p2 in edges1:
        for eq in hull2.equations:
            plane_normal = eq[:-1]
            plane_d = eq[-1]
            # Find a point on the plane
            plane_point = -plane_d * plane_normal
            
            intersection = _line_plane_intersection(p1, p2, plane_normal, plane_point)
            if intersection is not None:
                # Check if intersection point is inside box2
                inside = True
                for eq2 in hull2.equations:
                    if np.dot(eq2[:-1], intersection) + eq2[-1] > 1e-8:
                        inside = False
                        break
                if inside:
                    intersection_points.append(intersection)
    
    # Edges of box2 intersecting faces of box1
    for p1, p2 in edges2:
        for eq in hull1.equations:
            plane_normal = eq[:-1]
            plane_d = eq[-1]
            plane_point = -plane_d * plane_normal
            
            intersection = _line_plane_intersection(p1, p2, plane_normal, plane_point)
            if intersection is not None:
                # Check if intersection point is inside box1
                inside = True
                for eq2 in hull1.equations:
                    if np.dot(eq2[:-1], intersection) + eq2[-1] > 1e-8:
                        inside = False
                        break
                if inside:
                    intersection_points.append(intersection)
    
    # Compute intersection volume
    if len(intersection_points) < 4:
        return 0.0  # Not enough points for a volume
    
    intersection_points = np.array(intersection_points)
    
    # Remove duplicate points
    unique_points = []
    for pt in intersection_points:
        is_duplicate = False
        for upt in unique_points:
            if np.linalg.norm(pt - upt) < 1e-8:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(pt)
    
    if len(unique_points) < 4:
        return 0.0
    
    unique_points = np.array(unique_points)
    
    try:
        inter_hull = ConvexHull(unique_points)
        inter_vol = inter_hull.volume
    except Exception:
        return 0.0
    
    # Compute IoU
    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0
    
    iou = inter_vol / union_vol
    return float(np.clip(iou, 0.0, 1.0))


def compute_box3d_iou_aabb(box1_corners: np.ndarray, box2_corners: np.ndarray) -> float:
    """Compute approximate 3D IoU using axis-aligned bounding box approximation.
    
    This is faster but less accurate for rotated boxes. The intersection
    is computed using the AABB of each box, which overestimates IoU.
    
    Args:
        box1_corners: (8, 3) array of corner coordinates
        box2_corners: (8, 3) array of corner coordinates
    
    Returns:
        IoU value in [0, 1] (approximate, typically overestimated)
    """
    # Get axis-aligned bounding boxes
    min1, max1 = box1_corners.min(axis=0), box1_corners.max(axis=0)
    min2, max2 = box2_corners.min(axis=0), box2_corners.max(axis=0)
    
    # Compute AABB intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    
    # Check for no intersection
    if np.any(inter_max <= inter_min):
        return 0.0
    
    # AABB intersection volume
    inter_vol = np.prod(inter_max - inter_min)
    
    # Compute volumes using convex hull for accuracy
    try:
        vol1 = ConvexHull(box1_corners).volume
        vol2 = ConvexHull(box2_corners).volume
    except Exception:
        # Fallback to AABB volume
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)
    
    # Clamp intersection volume to not exceed actual box volumes
    # (AABB intersection can be larger than actual box volume for rotated boxes)
    inter_vol = min(inter_vol, vol1, vol2)
    
    # IoU
    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0
    
    iou = inter_vol / union_vol
    return float(np.clip(iou, 0.0, 1.0))


def compute_box3d_iou_matrix(
    pred_boxes: List[List[float]], 
    gt_boxes: List[List[float]],
    method: str = "precise"
) -> np.ndarray:
    """Compute pairwise 3D IoU between predictions and ground truths.
    
    Args:
        pred_boxes: List of N prediction bbox_3d params [x,y,z,w,h,l,r,p,y]
        gt_boxes: List of M ground truth bbox_3d params
        method: IoU computation method - "precise" (default) or "aabb" (faster approximation)
    
    Returns:
        iou_matrix: (N, M) array of IoU values
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))
    
    # Convert to corners
    pred_corners = np.array([bbox_3d_to_corners(b) for b in pred_boxes])
    gt_corners = np.array([bbox_3d_to_corners(b) for b in gt_boxes])
    
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    iou_matrix = np.zeros((n_pred, n_gt))
    
    # Select IoU computation function
    if method == "aabb":
        iou_func = compute_box3d_iou_aabb
    else:
        iou_func = compute_box3d_iou_precise
    
    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = iou_func(pred_corners[i], gt_corners[j])
    
    return iou_matrix


def benchmark_iou_methods(
    pred_boxes: List[List[float]], 
    gt_boxes: List[List[float]],
    num_samples: int = 100
) -> Dict[str, Any]:
    """Benchmark precise vs AABB IoU computation methods.
    
    Args:
        pred_boxes: List of prediction bbox_3d params
        gt_boxes: List of ground truth bbox_3d params
        num_samples: Number of box pairs to benchmark
    
    Returns:
        Dict with timing and accuracy comparison results
    """
    import time
    
    # Limit to num_samples pairs
    n_pred = min(len(pred_boxes), num_samples)
    n_gt = min(len(gt_boxes), num_samples)
    pred_boxes = pred_boxes[:n_pred]
    gt_boxes = gt_boxes[:n_gt]
    
    # Convert to corners
    pred_corners = np.array([bbox_3d_to_corners(b) for b in pred_boxes])
    gt_corners = np.array([bbox_3d_to_corners(b) for b in gt_boxes])
    
    # Benchmark precise method
    start = time.time()
    precise_ious = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            precise_ious[i, j] = compute_box3d_iou_precise(pred_corners[i], gt_corners[j])
    precise_time = time.time() - start
    
    # Benchmark AABB method
    start = time.time()
    aabb_ious = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            aabb_ious[i, j] = compute_box3d_iou_aabb(pred_corners[i], gt_corners[j])
    aabb_time = time.time() - start
    
    # Compute statistics
    diff = aabb_ious - precise_ious
    
    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_pairs": n_pred * n_gt,
        "precise_time_sec": precise_time,
        "aabb_time_sec": aabb_time,
        "speedup": precise_time / max(aabb_time, 1e-6),
        "mean_iou_precise": float(np.mean(precise_ious)),
        "mean_iou_aabb": float(np.mean(aabb_ious)),
        "mean_diff": float(np.mean(diff)),
        "max_diff": float(np.max(diff)),
        "std_diff": float(np.std(diff)),
        "correlation": float(np.corrcoef(precise_ious.flatten(), aabb_ious.flatten())[0, 1]) 
            if len(precise_ious.flatten()) > 1 else 1.0,
    }


def match_predictions_to_gt(
    ious: np.ndarray,
    iou_threshold: float,
    scores: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Match predictions to ground truths using greedy matching.
    
    Args:
        ious: (N, M) IoU matrix between N preds and M GTs
        iou_threshold: Minimum IoU for valid match
        scores: Optional (N,) prediction confidence scores for ordering
    
    Returns:
        dt_matches: (N,) indices of matched GT (-1 if unmatched)
        gt_matches: (M,) indices of matched pred (-1 if unmatched)
    """
    n_pred, n_gt = ious.shape
    dt_matches = np.full(n_pred, -1, dtype=np.int32)
    gt_matches = np.full(n_gt, -1, dtype=np.int32)
    
    if n_pred == 0 or n_gt == 0:
        return dt_matches, gt_matches
    
    # Order predictions by score (descending) if provided
    if scores is not None:
        pred_order = np.argsort(-scores)
    else:
        pred_order = np.arange(n_pred)
    
    # Greedy matching
    for pred_idx in pred_order:
        # Find best unmatched GT
        best_gt_idx = -1
        best_iou = iou_threshold
        
        for gt_idx in range(n_gt):
            if gt_matches[gt_idx] >= 0:
                continue  # Already matched
            if ious[pred_idx, gt_idx] >= best_iou:
                best_iou = ious[pred_idx, gt_idx]
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            dt_matches[pred_idx] = best_gt_idx
            gt_matches[best_gt_idx] = pred_idx
    
    return dt_matches, gt_matches


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
    
    Returns:
        AP value
    """
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # Add sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # Compute precision envelope (monotonically decreasing)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Find points where recall changes
    change_indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    
    # Sum (delta recall) * precision
    ap = np.sum((recalls[change_indices] - recalls[change_indices - 1]) * precisions[change_indices])
    
    return float(ap)


class Omni3DMetrics:
    """Official Omni3D AP3D/AR3D metrics calculator.
    
    This implements the COCO-style evaluation protocol adapted for 3D detection
    with IoU thresholds [0.15, 0.25, 0.50] as defined in the Omni3D benchmark.
    
    Uses precise 3D IoU computation via convex hull intersection (CPU).
    """
    
    def __init__(
        self,
        iou_thresholds: np.ndarray = IOU_THRESHOLDS_3D,
        max_dets: List[int] = MAX_DETS,
        iou_method: str = "precise"
    ):
        """Initialize Omni3D metrics calculator.
        
        Args:
            iou_thresholds: IoU thresholds for AP calculation
            max_dets: Max detections per image [1, 10, 100]
            iou_method: "precise" for exact convex hull intersection, 
                       "aabb" for faster axis-aligned approximation
        """
        self.iou_thresholds = iou_thresholds
        self.max_dets = max_dets
        self.iou_method_name = iou_method
        
        # Get method description
        if iou_method == "precise":
            self.iou_method_desc = "Precise 3D IoU (convex hull intersection)"
        else:
            self.iou_method_desc = "AABB approximation (faster, less accurate)"
        
        # Storage for evaluation data
        self.eval_data = []  # Per-image evaluation results
        self.category_map = {}  # category_name -> category_id
        
    def add_batch(
        self,
        image_id: str,
        category: str,
        predictions: List[Dict],
        gt_annotations: List[Dict],
        scores: Optional[List[float]] = None
    ):
        """Add predictions and ground truths for one image-category pair.
        
        Args:
            image_id: Unique image identifier
            category: Object category name
            predictions: List of {"bbox_3d": [...], "label": ...}
            gt_annotations: List of raw GT annotations
            scores: Optional confidence scores for predictions
        """
        if category not in self.category_map:
            self.category_map[category] = len(self.category_map)
        
        # Extract prediction boxes
        pred_boxes = [p["bbox_3d"] for p in predictions if "bbox_3d" in p]
        
        # Extract GT boxes from annotations
        gt_boxes = []
        for ann in gt_annotations:
            # Get center position (in camera coordinates)
            center = ann.get("center_cam", ann.get("bbox3D_cam", [0, 0, 0]))[:3]
            # Get dimensions
            dims = ann.get("dimensions", [1, 1, 1])
            # Get rotation (in radians, convert to degrees)
            R_cam = ann.get("R_cam", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            
            # Convert rotation matrix to Euler angles (simplified - just yaw for now)
            # Full Euler extraction would be more complex
            yaw = np.degrees(np.arctan2(R_cam[1][0], R_cam[0][0])) if R_cam else 0
            pitch = np.degrees(np.arcsin(-R_cam[2][0])) if R_cam else 0
            roll = np.degrees(np.arctan2(R_cam[2][1], R_cam[2][2])) if R_cam else 0
            
            gt_box = [
                center[0], center[1], center[2],
                dims[0], dims[1], dims[2],
                roll, pitch, yaw
            ]
            gt_boxes.append(gt_box)
        
        # Convert scores to numpy array
        if scores is None:
            scores_arr = np.ones(len(pred_boxes))
        else:
            scores_arr = np.array(scores[:len(pred_boxes)])
        
        # Compute IoU matrix
        if pred_boxes and gt_boxes:
            iou_matrix = compute_box3d_iou_matrix(pred_boxes, gt_boxes, method=self.iou_method_name)
        else:
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        
        self.eval_data.append({
            "image_id": image_id,
            "category": category,
            "category_id": self.category_map[category],
            "pred_boxes": pred_boxes,
            "gt_boxes": gt_boxes,
            "scores": scores_arr,
            "iou_matrix": iou_matrix,
            "n_pred": len(pred_boxes),
            "n_gt": len(gt_boxes),
        })
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute AP3D and AR3D metrics.
        
        Returns:
            Dict with metrics:
                - AP3D: Mean AP across all IoU thresholds
                - AP3D@0.15, AP3D@0.25, AP3D@0.50: AP at specific thresholds
                - AR3D@1, AR3D@10, AR3D@100: AR at max det limits
                - iou_method: String indicating which IoU computation was used
        """
        if not self.eval_data:
            return {
                "AP3D": 0.0,
                "AP3D@0.15": 0.0,
                "AP3D@0.25": 0.0,
                "AP3D@0.50": 0.0,
                "AR3D@1": 0.0,
                "AR3D@10": 0.0,
                "AR3D@100": 0.0,
                "iou_method": self.iou_method_desc,
            }
        
        metrics = {"iou_method": self.iou_method_desc}
        
        # Compute AP at each IoU threshold
        ap_per_threshold = []
        for iou_thresh in self.iou_thresholds:
            ap = self._compute_ap_at_threshold(iou_thresh)
            ap_per_threshold.append(ap)
            metrics[f"AP3D@{iou_thresh:.2f}"] = ap
        
        # Mean AP across thresholds
        metrics["AP3D"] = np.mean(ap_per_threshold)
        
        # Compute AR at different max detections
        for max_det in self.max_dets:
            ar = self._compute_ar_at_max_det(max_det)
            metrics[f"AR3D@{max_det}"] = ar
        
        return metrics
    
    def _compute_ap_at_threshold(self, iou_threshold: float) -> float:
        """Compute AP at a specific IoU threshold."""
        # Collect all predictions and matches across images
        all_scores = []
        all_matched = []  # 1 if matched, 0 if not
        total_gt = 0
        
        for data in self.eval_data:
            n_pred = data["n_pred"]
            n_gt = data["n_gt"]
            total_gt += n_gt
            
            if n_pred == 0:
                continue
            
            if n_gt == 0:
                # All predictions are false positives
                all_scores.extend(data["scores"].tolist())
                all_matched.extend([0] * n_pred)
                continue
            
            # Match predictions to GT
            dt_matches, _ = match_predictions_to_gt(
                data["iou_matrix"], iou_threshold, data["scores"]
            )
            
            all_scores.extend(data["scores"].tolist())
            all_matched.extend([1 if m >= 0 else 0 for m in dt_matches])
        
        if total_gt == 0 or len(all_scores) == 0:
            return 0.0
        
        # Sort by score descending
        all_scores = np.array(all_scores)
        all_matched = np.array(all_matched)
        sorted_idx = np.argsort(-all_scores)
        all_matched = all_matched[sorted_idx]
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(all_matched)
        fp_cumsum = np.cumsum(1 - all_matched)
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using 101-point interpolation
        return compute_ap(recalls, precisions)
    
    def _compute_ar_at_max_det(self, max_det: int) -> float:
        """Compute Average Recall at a specific max detection limit."""
        recalls_per_threshold = []
        
        for iou_threshold in self.iou_thresholds:
            total_tp = 0
            total_gt = 0
            
            for data in self.eval_data:
                n_pred = min(data["n_pred"], max_det)
                n_gt = data["n_gt"]
                total_gt += n_gt
                
                if n_pred == 0 or n_gt == 0:
                    continue
                
                # Truncate to max_det
                scores = data["scores"][:n_pred]
                iou_matrix = data["iou_matrix"][:n_pred]
                
                # Match predictions to GT
                dt_matches, _ = match_predictions_to_gt(
                    iou_matrix, iou_threshold, scores
                )
                
                total_tp += np.sum(dt_matches >= 0)
            
            if total_gt > 0:
                recalls_per_threshold.append(total_tp / total_gt)
            else:
                recalls_per_threshold.append(0.0)
        
        return np.mean(recalls_per_threshold) if recalls_per_threshold else 0.0
    
    def get_per_category_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics per category."""
        per_cat_metrics = {}
        
        for category in self.category_map.keys():
            # Filter data for this category
            cat_data = [d for d in self.eval_data if d["category"] == category]
            
            if not cat_data:
                continue
            
            # Create temporary evaluator for this category
            cat_evaluator = Omni3DMetrics(self.iou_thresholds, self.max_dets, self.iou_method_name)
            cat_evaluator.eval_data = cat_data
            cat_evaluator.category_map = {category: 0}
            
            per_cat_metrics[category] = cat_evaluator.compute_metrics()
        
        return per_cat_metrics
