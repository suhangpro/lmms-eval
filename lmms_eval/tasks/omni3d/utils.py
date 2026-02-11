"""
Core task functions for Omni3D 3D object detection benchmark.

This module provides the main interface between lmms-eval and the Omni3D dataset,
including data loading, prompt generation, result processing, and metric aggregation.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset
from loguru import logger as eval_logger
from PIL import Image

# Import local utilities
from lmms_eval.tasks.omni3d.eval_utils import (
    filter_annotation,
    parse_bbox_3d_from_text,
    validate_bbox_3d,
    Omni3DMetrics,
)

# Import configuration
from lmms_eval.tasks.omni3d.config import (
    IMAGE_ROOT,
    JSON_ROOT,
    CATEGORY_META_PATH,
)

# Load category metadata once at module import
_CATEGORY_META = None


def get_category_metadata():
    """Load and cache category metadata from local file shipped with the code."""
    global _CATEGORY_META
    if _CATEGORY_META is None:
        if not os.path.exists(CATEGORY_META_PATH):
            raise FileNotFoundError(
                f"Category metadata not found at {CATEGORY_META_PATH}. "
                f"This file should be shipped with the lmms-eval package."
            )
        with open(CATEGORY_META_PATH, "r") as f:
            _CATEGORY_META = json.load(f)
    return _CATEGORY_META


def load_omni3d_dataset(dataset_name: str, split: str = "test") -> Dict[str, Any]:
    """Load Omni3D dataset from local JSON file.
    
    Args:
        dataset_name: One of "KITTI", "nuScenes", "SUNRGBD", "Objectron", 
                      "ARKitScenes", "Hypersim"
        split: "train", "val", or "test"
    
    Returns:
        Dict with keys: images, annotations, imgid2info, categories
    """
    json_path = os.path.join(JSON_ROOT, f"{dataset_name}_{split}.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")
    
    eval_logger.info(f"Loading {dataset_name} {split} from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Create image_id to image_info mapping
    imgid2info = {img["id"]: img for img in data["images"]}
    
    eval_logger.info(
        f"Loaded {len(data['images'])} images, {len(data['annotations'])} annotations"
    )
    
    return {
        "images": data["images"],
        "annotations": data["annotations"],
        "imgid2info": imgid2info,
        "categories": data.get("categories", []),
    }


def build_omni3d_instances(
    dataset_name: str, 
    split: str = "test",
    category_filter: str = None
) -> List[Dict]:
    """Build task instances from Omni3D dataset.
    
    Strategy: Create one instance per (image, category) pair.
    This allows the model to focus on one category at a time.
    
    Args:
        dataset_name: Name of dataset (e.g., "Objectron")
        split: Data split ("test", "train", etc.)
        category_filter: If provided, only include this category
    
    Returns:
        List of document dicts for lmms-eval
    """
    data = load_omni3d_dataset(dataset_name, split)
    imgid2info = data["imgid2info"]
    annotations = data["annotations"]
    category_meta = get_category_metadata()
    thing_classes = category_meta["thing_classes"]
    
    # Apply category filter if specified
    if category_filter:
        eval_logger.info(f"Filtering to category: {category_filter}")
        thing_classes = [category_filter] if category_filter in thing_classes else []
    
    # Filter settings (from Omni3D official)
    filter_settings = {
        "visibility_thres": 0.33333333,
        "truncation_thres": 0.33333333,
        "min_height_thres": 0.0625,
        "max_height_thres": 1.5,
        "max_depth": 1e8,
        "trunc_2D_boxes": True,
        "modal_2D_boxes": False,
    }
    
    # Group annotations by (image_id, category_name)
    image_category_pairs = {}
    
    for ann in annotations:
        # Skip filtered annotations
        img_info = imgid2info[ann["image_id"]]
        if filter_annotation(ann, img_info, filter_settings):
            continue
        
        # Get category name
        category_name = ann.get("category_name", "")
        if category_name not in thing_classes:
            continue
        
        key = (ann["image_id"], category_name)
        if key not in image_category_pairs:
            image_category_pairs[key] = {
                "image_id": ann["image_id"],
                "category": category_name,
                "gt_annotations": [],
            }
        image_category_pairs[key]["gt_annotations"].append(ann)
    
    # Build document list
    instances = []
    for (image_id, category), pair_data in image_category_pairs.items():
        img_info = imgid2info[image_id]
        
        doc = {
            "index": f"{dataset_name}_{split}_{image_id}_{category}",
            "image_id": image_id,
            "category": category,
            "image_path": img_info["file_path"],
            "width": img_info["width"],
            "height": img_info["height"],
            "K": img_info["K"],  # Camera intrinsics
            "num_gt_objects": len(pair_data["gt_annotations"]),
            "dataset_name": dataset_name,
            "split": split,
        }
        instances.append(doc)
    
    eval_logger.info(
        f"Built {len(instances)} instances from {len(image_category_pairs)} "
        f"(image, category) pairs"
    )
    return instances


def omni3d_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Load image for Omni3D task.
    
    Args:
        doc: Document dict with 'image_path' field
    
    Returns:
        List containing single PIL Image in RGB
    """
    image_path = os.path.join(IMAGE_ROOT, doc["image_path"])
    
    if not os.path.exists(image_path):
        # Try without leading path components (for Objectron which has different structure)
        alt_path = os.path.join(IMAGE_ROOT, os.path.basename(doc["image_path"]))
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    return [image]


def omni3d_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] = None
) -> str:
    """Generate prompt for 3D object detection.
    
    Args:
        doc: Document with category, K (camera intrinsics), width, height
        lmms_eval_specific_kwargs: Optional prompt customization
    
    Returns:
        Formatted prompt string
    """
    category = doc["category"]
    
    # Simplified prompt matching VLMEvalKit's approach
    # Original VLMEvalKit prompt doesn't include camera intrinsics - keeps it simple
    prompt = f"""Locate the {category} in the provided image and output their positions and dimensions using 3D bounding boxes. The results must be in the JSON format: `[{{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"{category}"}}]`. If no {category} is visible, output `[]`."""
    
    return prompt


def omni3d_process_results(
    doc: Dict[str, Any], results: List[str]
) -> Dict[str, Any]:
    """Process model predictions for Omni3D.
    
    Args:
        doc: Document dict with ground truth info
        results: List of model response strings
    
    Returns:
        Dict with metrics for aggregation
    """
    response = results[0] if results else ""
    
    # Parse predictions
    pred_bboxes = parse_bbox_3d_from_text(response)
    
    # Validate predictions
    valid_bboxes = [
        bbox for bbox in pred_bboxes if validate_bbox_3d(bbox.get("bbox_3d", []))
    ]
    
    # Basic metrics (without IoU computation)
    num_predictions = len(pred_bboxes)
    num_valid = len(valid_bboxes)
    num_gt = doc.get("num_gt_objects", 0)
    
    # Load GT annotations for AP3D computation
    gt_annotations = _get_gt_annotations_for_doc(doc)
    
    result = {
        "image_id": doc["image_id"],
        "category": doc["category"],
        "dataset_name": doc["dataset_name"],
        "num_predictions": num_predictions,
        "num_valid_predictions": num_valid,
        "num_gt_objects": num_gt,
        "has_valid_prediction": num_valid > 0,
        "response": response,
        "predictions": valid_bboxes,
        "gt_annotations": gt_annotations,  # For AP3D computation
    }
    
    return {
        "omni3d_valid_rate": result, 
        "omni3d_recall_proxy": result,
        "omni3d_ap3d": result,
        "omni3d_ap3d_15": result,
        "omni3d_ap3d_25": result,
        "omni3d_ap3d_50": result,
        "omni3d_ar3d_1": result,
        "omni3d_ar3d_10": result,
        "omni3d_ar3d_100": result,
    }


# Cache for GT annotations by (dataset, image_id, category)
_GT_ANNOTATION_CACHE = {}


def _get_gt_annotations_for_doc(doc: Dict[str, Any]) -> List[Dict]:
    """Load GT annotations for a document.
    
    Args:
        doc: Document dict with dataset_name, split, image_id, category
    
    Returns:
        List of GT annotation dicts
    """
    dataset_name = doc["dataset_name"]
    split = doc.get("split", "test")
    image_id = doc["image_id"]
    category = doc["category"]
    
    cache_key = f"{dataset_name}_{split}"
    
    # Load dataset if not cached
    if cache_key not in _GT_ANNOTATION_CACHE:
        try:
            data = load_omni3d_dataset(dataset_name, split)
            # Index annotations by (image_id, category)
            ann_index = defaultdict(list)
            for ann in data["annotations"]:
                key = (ann["image_id"], ann.get("category_name", ""))
                ann_index[key].append(ann)
            _GT_ANNOTATION_CACHE[cache_key] = {
                "ann_index": ann_index,
                "imgid2info": data["imgid2info"],
            }
        except Exception as e:
            eval_logger.warning(f"Failed to load GT annotations: {e}")
            return []
    
    cached = _GT_ANNOTATION_CACHE[cache_key]
    ann_index = cached["ann_index"]
    imgid2info = cached["imgid2info"]
    
    # Get annotations for this (image_id, category)
    raw_annotations = ann_index.get((image_id, category), [])
    
    # Filter using Omni3D criteria
    filter_settings = {
        "visibility_thres": 0.33333333,
        "truncation_thres": 0.33333333,
        "min_height_thres": 0.0625,
        "max_height_thres": 1.5,
        "max_depth": 1e8,
        "trunc_2D_boxes": True,
        "modal_2D_boxes": False,
    }
    
    img_info = imgid2info.get(image_id, {"width": 1, "height": 1})
    filtered_annotations = [
        ann for ann in raw_annotations 
        if not filter_annotation(ann, img_info, filter_settings)
    ]
    
    return filtered_annotations


def omni3d_aggregate_results(results: List[Dict]) -> float:
    """Aggregate Omni3D evaluation results.
    
    This provides basic metrics without expensive 3D IoU computation.
    For official metrics (AP3D, AR3D), use the external evaluation script.
    
    Args:
        results: List of result dicts from process_results
    
    Returns:
        Overall valid prediction rate
    """
    if not results:
        eval_logger.warning("No results to aggregate")
        return 0.0
    
    total_predictions = sum(r["num_predictions"] for r in results)
    total_valid = sum(r["num_valid_predictions"] for r in results)
    total_gt = sum(r["num_gt_objects"] for r in results)
    total_with_predictions = sum(1 for r in results if r["has_valid_prediction"])
    
    # Per-category stats
    category_stats = defaultdict(
        lambda: {"pred": 0, "valid": 0, "gt": 0, "count": 0}
    )
    for r in results:
        cat = r["category"]
        category_stats[cat]["pred"] += r["num_predictions"]
        category_stats[cat]["valid"] += r["num_valid_predictions"]
        category_stats[cat]["gt"] += r["num_gt_objects"]
        category_stats[cat]["count"] += 1
    
    # Per-dataset stats
    dataset_stats = defaultdict(lambda: {"pred": 0, "valid": 0, "gt": 0, "count": 0})
    for r in results:
        ds = r["dataset_name"]
        dataset_stats[ds]["pred"] += r["num_predictions"]
        dataset_stats[ds]["valid"] += r["num_valid_predictions"]
        dataset_stats[ds]["gt"] += r["num_gt_objects"]
        dataset_stats[ds]["count"] += 1
    
    valid_rate = total_valid / total_predictions if total_predictions > 0 else 0.0
    recall_proxy = total_with_predictions / len(results) if results else 0.0
    
    eval_logger.info("=" * 80)
    eval_logger.info("Omni3D Evaluation Results (Basic Metrics)")
    eval_logger.info("=" * 80)
    eval_logger.info(f"Total samples: {len(results)}")
    eval_logger.info(f"Total predictions: {total_predictions}")
    eval_logger.info(f"Valid predictions: {total_valid} ({valid_rate:.2%})")
    eval_logger.info(f"Total GT objects: {total_gt}")
    eval_logger.info(
        f"Samples with predictions: {total_with_predictions} ({recall_proxy:.2%})"
    )
    eval_logger.info("")
    
    eval_logger.info("Per-Category Statistics:")
    eval_logger.info("-" * 80)
    eval_logger.info(
        f"{'Category':<20} {'Samples':>8} {'Pred':>8} {'Valid':>8} {'GT':>8} {'Valid%':>8}"
    )
    eval_logger.info("-" * 80)
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        vr = stats["valid"] / stats["pred"] if stats["pred"] > 0 else 0
        eval_logger.info(
            f"{cat:<20} {stats['count']:>8} {stats['pred']:>8} "
            f"{stats['valid']:>8} {stats['gt']:>8} {vr:>7.1%}"
        )
    
    eval_logger.info("")
    eval_logger.info("Per-Dataset Statistics:")
    eval_logger.info("-" * 80)
    for ds in sorted(dataset_stats.keys()):
        stats = dataset_stats[ds]
        vr = stats["valid"] / stats["pred"] if stats["pred"] > 0 else 0
        eval_logger.info(
            f"{ds}: {stats['count']} samples, {stats['valid']}/{stats['pred']} valid ({vr:.1%})"
        )
    
    eval_logger.info("=" * 80)
    
    return valid_rate


def omni3d_recall_proxy_agg(results: List[Dict]) -> float:
    """Aggregate recall proxy metric (% of samples with any valid prediction)."""
    if not results:
        return 0.0
    total_with_predictions = sum(1 for r in results if r["has_valid_prediction"])
    return total_with_predictions / len(results)


# Cache for computed AP3D metrics to avoid recomputation
_AP3D_METRICS_CACHE: Dict[int, Dict] = {}


def _compute_ap3d_metrics(results: List[Dict]) -> Dict:
    """Compute all AP3D/AR3D metrics (cached).
    
    Args:
        results: List of result dicts from process_results
    
    Returns:
        Dict with all metrics (AP3D, AP3D@0.15, etc.)
    """
    # Use hash of result IDs as cache key
    cache_key = hash(tuple(r.get("image_id", i) for i, r in enumerate(results)))
    
    if cache_key in _AP3D_METRICS_CACHE:
        return _AP3D_METRICS_CACHE[cache_key]
    
    if not results:
        eval_logger.warning("No results for AP3D computation")
        empty_metrics = {
            "AP3D": 0.0,
            "AP3D@0.15": 0.0,
            "AP3D@0.25": 0.0,
            "AP3D@0.50": 0.0,
            "AR3D@1": 0.0,
            "AR3D@10": 0.0,
            "AR3D@100": 0.0,
        }
        _AP3D_METRICS_CACHE[cache_key] = empty_metrics
        return empty_metrics
    
    # Initialize metrics calculator (precise 3D IoU via convex hull intersection)
    metrics_calculator = Omni3DMetrics(iou_method="precise")
    
    # Log IoU computation method
    eval_logger.info(f"3D IoU Method: {metrics_calculator.iou_method_desc}")
    
    # Add all predictions to the calculator
    for r in results:
        metrics_calculator.add_batch(
            image_id=str(r["image_id"]),
            category=r["category"],
            predictions=r.get("predictions", []),
            gt_annotations=r.get("gt_annotations", []),
            scores=None  # Assume uniform confidence
        )
    
    # Compute metrics
    metrics = metrics_calculator.compute_metrics()
    
    # Log detailed results
    eval_logger.info("=" * 80)
    eval_logger.info("Omni3D Official AP3D Metrics (3D IoU)")
    eval_logger.info(f"  IoU Method: {metrics.get('iou_method', 'unknown')}")
    eval_logger.info("=" * 80)
    eval_logger.info(f"  AP3D (mean):    {metrics['AP3D']:.4f}")
    eval_logger.info(f"  AP3D@0.15:      {metrics.get('AP3D@0.15', 0):.4f}")
    eval_logger.info(f"  AP3D@0.25:      {metrics.get('AP3D@0.25', 0):.4f}")
    eval_logger.info(f"  AP3D@0.50:      {metrics.get('AP3D@0.50', 0):.4f}")
    eval_logger.info("-" * 80)
    eval_logger.info(f"  AR3D@1:         {metrics.get('AR3D@1', 0):.4f}")
    eval_logger.info(f"  AR3D@10:        {metrics.get('AR3D@10', 0):.4f}")
    eval_logger.info(f"  AR3D@100:       {metrics.get('AR3D@100', 0):.4f}")
    eval_logger.info("=" * 80)
    
    # Log per-category metrics
    per_cat = metrics_calculator.get_per_category_metrics()
    if per_cat:
        eval_logger.info("Per-Category AP3D:")
        eval_logger.info("-" * 80)
        for cat, cat_metrics in sorted(per_cat.items()):
            eval_logger.info(f"  {cat:<20}: AP3D={cat_metrics['AP3D']:.4f}")
        eval_logger.info("=" * 80)
    
    # Cache and return
    _AP3D_METRICS_CACHE[cache_key] = metrics
    return metrics


def omni3d_ap3d_agg(results: List[Dict]) -> float:
    """Aggregate official AP3D metric (mean over thresholds)."""
    return _compute_ap3d_metrics(results).get("AP3D", 0.0)


def omni3d_ap3d_15_agg(results: List[Dict]) -> float:
    """Aggregate AP3D@0.15 metric."""
    return _compute_ap3d_metrics(results).get("AP3D@0.15", 0.0)


def omni3d_ap3d_25_agg(results: List[Dict]) -> float:
    """Aggregate AP3D@0.25 metric."""
    return _compute_ap3d_metrics(results).get("AP3D@0.25", 0.0)


def omni3d_ap3d_50_agg(results: List[Dict]) -> float:
    """Aggregate AP3D@0.50 metric."""
    return _compute_ap3d_metrics(results).get("AP3D@0.50", 0.0)


def omni3d_ar3d_1_agg(results: List[Dict]) -> float:
    """Aggregate AR3D@1 metric."""
    return _compute_ap3d_metrics(results).get("AR3D@1", 0.0)


def omni3d_ar3d_10_agg(results: List[Dict]) -> float:
    """Aggregate AR3D@10 metric."""
    return _compute_ap3d_metrics(results).get("AR3D@10", 0.0)


def omni3d_ar3d_100_agg(results: List[Dict]) -> float:
    """Aggregate AR3D@100 metric."""
    return _compute_ap3d_metrics(results).get("AR3D@100", 0.0)


# Dataset loading functions for lmms-eval
# Cache for dataset instances to avoid reloading
_DATASET_CACHE = {}


# ============================================================================
# Objectron process_docs functions
# ============================================================================
def omni3d_process_docs_objectron(dataset):
    """Process documents for Objectron dataset (all categories)."""
    return _build_omni3d_dataset("Objectron", "test", None)


def omni3d_process_docs_objectron_cup(dataset):
    """Process documents for Objectron - cup only."""
    return _build_omni3d_dataset("Objectron", "test", "cup")


def omni3d_process_docs_objectron_chair(dataset):
    """Process documents for Objectron - chair only."""
    return _build_omni3d_dataset("Objectron", "test", "chair")


def omni3d_process_docs_objectron_laptop(dataset):
    """Process documents for Objectron - laptop only."""
    return _build_omni3d_dataset("Objectron", "test", "laptop")


# ============================================================================
# ARKitScenes process_docs functions
# ============================================================================
def omni3d_process_docs_arkitscenes(dataset):
    """Process documents for ARKitScenes dataset (all categories)."""
    return _build_omni3d_dataset("ARKitScenes", "test", None)


def omni3d_process_docs_arkitscenes_chair(dataset):
    """Process documents for ARKitScenes - chair only."""
    return _build_omni3d_dataset("ARKitScenes", "test", "chair")


def omni3d_process_docs_arkitscenes_cabinet(dataset):
    """Process documents for ARKitScenes - cabinet only."""
    return _build_omni3d_dataset("ARKitScenes", "test", "cabinet")


# ============================================================================
# Hypersim process_docs functions
# ============================================================================
def omni3d_process_docs_hypersim(dataset):
    """Process documents for Hypersim dataset (all categories)."""
    return _build_omni3d_dataset("Hypersim", "test", None)


def omni3d_process_docs_hypersim_chair(dataset):
    """Process documents for Hypersim - chair only."""
    return _build_omni3d_dataset("Hypersim", "test", "chair")


# ============================================================================
# KITTI process_docs functions
# ============================================================================
def omni3d_process_docs_kitti(dataset):
    """Process documents for KITTI dataset (all categories)."""
    return _build_omni3d_dataset("KITTI", "test", None)


def omni3d_process_docs_kitti_car(dataset):
    """Process documents for KITTI - car only."""
    return _build_omni3d_dataset("KITTI", "test", "car")


def omni3d_process_docs_kitti_pedestrian(dataset):
    """Process documents for KITTI - pedestrian only."""
    return _build_omni3d_dataset("KITTI", "test", "pedestrian")


# ============================================================================
# nuScenes process_docs functions
# ============================================================================
def omni3d_process_docs_nuscenes(dataset):
    """Process documents for nuScenes dataset (all categories)."""
    return _build_omni3d_dataset("nuScenes", "test", None)


def omni3d_process_docs_nuscenes_car(dataset):
    """Process documents for nuScenes - car only."""
    return _build_omni3d_dataset("nuScenes", "test", "car")


def omni3d_process_docs_nuscenes_pedestrian(dataset):
    """Process documents for nuScenes - pedestrian only."""
    return _build_omni3d_dataset("nuScenes", "test", "pedestrian")


# ============================================================================
# SUNRGBD process_docs functions
# ============================================================================
def omni3d_process_docs_sunrgbd(dataset):
    """Process documents for SUNRGBD dataset (all categories)."""
    return _build_omni3d_dataset("SUNRGBD", "test", None)


def omni3d_process_docs_sunrgbd_chair(dataset):
    """Process documents for SUNRGBD - chair only."""
    return _build_omni3d_dataset("SUNRGBD", "test", "chair")


# Legacy alias for backward compatibility
def omni3d_process_docs(dataset):
    """Legacy: Process documents for Objectron (default)."""
    return omni3d_process_docs_objectron(dataset)


def omni3d_process_docs_cup(dataset):
    """Legacy: cup category."""
    return omni3d_process_docs_objectron_cup(dataset)


def omni3d_process_docs_chair(dataset):
    """Legacy: chair category."""
    return omni3d_process_docs_objectron_chair(dataset)


def omni3d_process_docs_laptop(dataset):
    """Legacy: laptop category."""
    return omni3d_process_docs_objectron_laptop(dataset)


def _build_omni3d_dataset(dataset_name: str, split: str, category_filter: str = None):
    """Build Omni3D dataset with optional category filter.
    
    Returns:
        HuggingFace Dataset object with Omni3D instances
    """
    cache_key = f"{dataset_name}_{split}_{category_filter or 'all'}"
    
    if cache_key not in _DATASET_CACHE:
        eval_logger.info(f"Building Omni3D instances for {dataset_name} {split}...")
        if category_filter:
            eval_logger.info(f"Filtering to category: {category_filter}")
        instances = build_omni3d_instances(dataset_name, split, category_filter)
        
        if not instances:
            eval_logger.warning(f"No instances found!")
            _DATASET_CACHE[cache_key] = Dataset.from_dict({
                "index": [], "image_id": [], "category": [], "image_path": [],
                "width": [], "height": [], "K": [], "num_gt_objects": [],
                "dataset_name": [], "split": []
            })
        else:
            _DATASET_CACHE[cache_key] = Dataset.from_dict({
                key: [inst[key] for inst in instances]
                for key in instances[0].keys()
            })
        eval_logger.info(f"Created Dataset with {len(_DATASET_CACHE[cache_key])} instances")
    
    return _DATASET_CACHE[cache_key]


def _build_omni3d_dataset_sampled(
    dataset_name: str, 
    split: str, 
    category_filter: str = None,
    num_samples: int = 10,
    sampling: str = "even"
):
    """Build Omni3D dataset with sampling.
    
    Args:
        dataset_name: Name of the dataset
        split: Data split
        category_filter: Optional category to filter
        num_samples: Number of samples to select
        sampling: Sampling strategy - "even" for evenly spaced, "random" for random
    
    Returns:
        HuggingFace Dataset object with sampled instances
    """
    import random
    
    cache_key = f"{dataset_name}_{split}_{category_filter or 'all'}_{sampling}_{num_samples}"
    
    if cache_key not in _DATASET_CACHE:
        eval_logger.info(f"Building sampled Omni3D instances for {dataset_name} {split}...")
        eval_logger.info(f"Sampling: {num_samples} samples, strategy: {sampling}")
        
        if category_filter:
            eval_logger.info(f"Filtering to category: {category_filter}")
        
        instances = build_omni3d_instances(dataset_name, split, category_filter)
        
        if not instances:
            eval_logger.warning(f"No instances found!")
            _DATASET_CACHE[cache_key] = Dataset.from_dict({
                "index": [], "image_id": [], "category": [], "image_path": [],
                "width": [], "height": [], "K": [], "num_gt_objects": [],
                "dataset_name": [], "split": []
            })
        else:
            # Apply sampling
            total = len(instances)
            if num_samples >= total:
                sampled = instances
                eval_logger.info(f"Requested {num_samples} but only {total} available, using all")
            elif sampling == "even":
                # Evenly spaced sampling
                indices = [int(i * total / num_samples) for i in range(num_samples)]
                sampled = [instances[i] for i in indices]
                eval_logger.info(f"Evenly sampled {num_samples} from {total} (indices: {indices[:5]}...)")
            elif sampling == "random":
                random.seed(42)
                sampled = random.sample(instances, num_samples)
                eval_logger.info(f"Randomly sampled {num_samples} from {total}")
            else:
                # Default to first N
                sampled = instances[:num_samples]
                eval_logger.info(f"Selected first {num_samples} from {total}")
            
            _DATASET_CACHE[cache_key] = Dataset.from_dict({
                key: [inst[key] for inst in sampled]
                for key in sampled[0].keys()
            })
        
        eval_logger.info(f"Created Dataset with {len(_DATASET_CACHE[cache_key])} instances")
    
    return _DATASET_CACHE[cache_key]


# ============================================================================
# Toy/Debug tasks - small sampled datasets for quick testing
# ============================================================================
def omni3d_process_docs_arkitscenes_toy(dataset):
    """ARKitScenes toy task: 10 evenly spaced samples."""
    return _build_omni3d_dataset_sampled("ARKitScenes", "test", None, num_samples=10, sampling="even")


def omni3d_process_docs_hypersim_toy(dataset):
    """Hypersim toy task: 10 evenly spaced samples."""
    return _build_omni3d_dataset_sampled("Hypersim", "test", None, num_samples=10, sampling="even")


def omni3d_process_docs_objectron_toy(dataset):
    """Objectron toy task: 10 evenly spaced samples."""
    return _build_omni3d_dataset_sampled("Objectron", "test", None, num_samples=10, sampling="even")


def omni3d_process_docs_kitti_toy(dataset):
    """KITTI toy task: 10 evenly spaced samples."""
    return _build_omni3d_dataset_sampled("KITTI", "test", None, num_samples=10, sampling="even")


def omni3d_process_docs_nuscenes_toy(dataset):
    """nuScenes toy task: 10 evenly spaced samples."""
    return _build_omni3d_dataset_sampled("nuScenes", "test", None, num_samples=10, sampling="even")


def omni3d_process_docs_sunrgbd_toy(dataset):
    """SUNRGBD toy task: 10 evenly spaced samples."""
    return _build_omni3d_dataset_sampled("SUNRGBD", "test", None, num_samples=10, sampling="even")
