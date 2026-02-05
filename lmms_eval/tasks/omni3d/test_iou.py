#!/usr/bin/env python3
"""Test and benchmark 3D IoU implementations."""

import numpy as np
import time
import sys

# Add parent directory to path
sys.path.insert(0, "/home/hangsu/Space1/lmms-eval")

from lmms_eval.tasks.omni3d.eval_utils import (
    bbox_3d_to_corners,
    compute_box3d_iou_precise,
    compute_box3d_iou_aabb,
    compute_box3d_iou_matrix,
    benchmark_iou_methods,
)


def generate_random_box(center_range=5.0, size_range=(0.5, 3.0), rotation_range=45.0):
    """Generate a random 3D bounding box."""
    center = np.random.uniform(-center_range, center_range, 3)
    center[2] = abs(center[2]) + 1.0  # Ensure positive depth
    
    sizes = np.random.uniform(size_range[0], size_range[1], 3)
    rotations = np.random.uniform(-rotation_range, rotation_range, 3)
    
    return list(center) + list(sizes) + list(rotations)


def test_identical_boxes():
    """Test that identical boxes have IoU = 1.0."""
    print("\n=== Test: Identical Boxes ===")
    box = [0, 0, 5, 1, 2, 3, 0, 0, 0]
    corners = bbox_3d_to_corners(box)
    
    iou_precise = compute_box3d_iou_precise(corners, corners)
    iou_aabb = compute_box3d_iou_aabb(corners, corners)
    
    print(f"Precise IoU: {iou_precise:.4f} (expected: 1.0)")
    print(f"AABB IoU: {iou_aabb:.4f} (expected: 1.0)")
    
    assert abs(iou_precise - 1.0) < 0.01, f"Precise IoU for identical boxes should be ~1.0, got {iou_precise}"
    assert abs(iou_aabb - 1.0) < 0.01, f"AABB IoU for identical boxes should be ~1.0, got {iou_aabb}"
    print("✓ PASSED")


def test_non_overlapping_boxes():
    """Test that non-overlapping boxes have IoU = 0.0."""
    print("\n=== Test: Non-Overlapping Boxes ===")
    box1 = [0, 0, 5, 1, 1, 1, 0, 0, 0]
    box2 = [10, 10, 5, 1, 1, 1, 0, 0, 0]  # Far away
    
    corners1 = bbox_3d_to_corners(box1)
    corners2 = bbox_3d_to_corners(box2)
    
    iou_precise = compute_box3d_iou_precise(corners1, corners2)
    iou_aabb = compute_box3d_iou_aabb(corners1, corners2)
    
    print(f"Precise IoU: {iou_precise:.4f} (expected: 0.0)")
    print(f"AABB IoU: {iou_aabb:.4f} (expected: 0.0)")
    
    assert iou_precise == 0.0, f"Precise IoU for non-overlapping boxes should be 0.0, got {iou_precise}"
    assert iou_aabb == 0.0, f"AABB IoU for non-overlapping boxes should be 0.0, got {iou_aabb}"
    print("✓ PASSED")


def test_partial_overlap():
    """Test partially overlapping boxes."""
    print("\n=== Test: Partial Overlap (Axis-Aligned) ===")
    box1 = [0, 0, 5, 2, 2, 2, 0, 0, 0]
    box2 = [1, 0, 5, 2, 2, 2, 0, 0, 0]  # Shifted by 1 along x
    
    corners1 = bbox_3d_to_corners(box1)
    corners2 = bbox_3d_to_corners(box2)
    
    # For axis-aligned boxes of size 2x2x2, shifted by 1 along x:
    # Overlap = 1 x 2 x 2 = 4
    # Union = 2*8 - 4 = 12
    # Expected IoU = 4/12 = 0.333...
    expected_iou = 4.0 / 12.0
    
    iou_precise = compute_box3d_iou_precise(corners1, corners2)
    iou_aabb = compute_box3d_iou_aabb(corners1, corners2)
    
    print(f"Precise IoU: {iou_precise:.4f} (expected: ~{expected_iou:.4f})")
    print(f"AABB IoU: {iou_aabb:.4f} (expected: ~{expected_iou:.4f})")
    
    # For axis-aligned boxes, both methods should be accurate
    assert abs(iou_precise - expected_iou) < 0.05, f"Precise IoU error too large"
    assert abs(iou_aabb - expected_iou) < 0.05, f"AABB IoU error too large"
    print("✓ PASSED")


def test_rotated_boxes():
    """Test rotated boxes where AABB should differ from precise."""
    print("\n=== Test: Rotated Boxes (AABB vs Precise) ===")
    box1 = [0, 0, 5, 2, 2, 2, 0, 0, 0]
    box2 = [0, 0, 5, 2, 2, 2, 0, 0, 45]  # Same but rotated 45 degrees around z
    
    corners1 = bbox_3d_to_corners(box1)
    corners2 = bbox_3d_to_corners(box2)
    
    iou_precise = compute_box3d_iou_precise(corners1, corners2)
    iou_aabb = compute_box3d_iou_aabb(corners1, corners2)
    
    print(f"Precise IoU: {iou_precise:.4f}")
    print(f"AABB IoU: {iou_aabb:.4f}")
    print(f"Difference: {iou_aabb - iou_precise:.4f}")
    
    # AABB typically overestimates for rotated boxes
    # The precise IoU should be less than or equal to AABB
    assert iou_aabb >= iou_precise - 0.01, "AABB should not underestimate compared to precise"
    print("✓ PASSED (AABB overestimates as expected)")


def test_matrix_computation():
    """Test batch IoU matrix computation."""
    print("\n=== Test: Matrix Computation ===")
    pred_boxes = [
        [0, 0, 5, 1, 1, 1, 0, 0, 0],
        [1, 1, 5, 1, 1, 1, 0, 0, 0],
    ]
    gt_boxes = [
        [0, 0, 5, 1, 1, 1, 0, 0, 0],
        [0.5, 0.5, 5, 1, 1, 1, 0, 0, 0],
    ]
    
    iou_precise = compute_box3d_iou_matrix(pred_boxes, gt_boxes, method="precise")
    iou_aabb = compute_box3d_iou_matrix(pred_boxes, gt_boxes, method="aabb")
    
    print(f"Precise IoU matrix:\n{iou_precise}")
    print(f"AABB IoU matrix:\n{iou_aabb}")
    
    # First pred should match first gt exactly
    assert iou_precise[0, 0] > 0.99, f"Expected IoU ~1.0 for identical boxes, got {iou_precise[0, 0]}"
    print("✓ PASSED")


def benchmark_random_boxes(n_boxes=20):
    """Benchmark with random boxes."""
    print(f"\n=== Benchmark: {n_boxes}x{n_boxes} Random Box Pairs ===")
    
    np.random.seed(42)
    pred_boxes = [generate_random_box() for _ in range(n_boxes)]
    gt_boxes = [generate_random_box() for _ in range(n_boxes)]
    
    results = benchmark_iou_methods(pred_boxes, gt_boxes, num_samples=n_boxes)
    
    print(f"Number of pairs: {results['n_pairs']}")
    print(f"Precise time: {results['precise_time_sec']:.3f}s")
    print(f"AABB time: {results['aabb_time_sec']:.3f}s")
    print(f"Speedup (AABB vs Precise): {results['speedup']:.1f}x")
    print(f"Mean IoU (Precise): {results['mean_iou_precise']:.4f}")
    print(f"Mean IoU (AABB): {results['mean_iou_aabb']:.4f}")
    print(f"Mean difference (AABB - Precise): {results['mean_diff']:.4f}")
    print(f"Max difference: {results['max_diff']:.4f}")
    print(f"Std difference: {results['std_diff']:.4f}")
    print(f"Correlation: {results['correlation']:.4f}")
    
    return results


def main():
    print("=" * 60)
    print("3D IoU Implementation Test Suite")
    print("=" * 60)
    
    # Run tests
    test_identical_boxes()
    test_non_overlapping_boxes()
    test_partial_overlap()
    test_rotated_boxes()
    test_matrix_computation()
    
    # Benchmark
    results = benchmark_random_boxes(n_boxes=20)
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    print("\n=== Summary ===")
    print("The precise 3D IoU implementation uses convex hull intersection")
    print("to compute exact overlap volume between oriented bounding boxes.")
    print(f"AABB approximation is ~{results['speedup']:.1f}x faster but overestimates")
    print(f"IoU by an average of {results['mean_diff']:.4f} (max: {results['max_diff']:.4f})")


if __name__ == "__main__":
    main()
