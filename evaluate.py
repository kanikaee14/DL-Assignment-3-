"""
Evaluation script using COCO metrics (AP50).
For local validation before submission.
"""

import os
import json
import numpy as np
import tifffile
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_coco_annotations(data_dir, image_id_mapping):
    """
    Create COCO format annotations from dataset masks.
    
    Args:
        data_dir: Path to dataset directory
        image_id_mapping: Dict mapping filename to image_id
    
    Returns:
        COCO format dict
    """
    images = []
    annotations = []
    annotation_id = 1
    
    train_dir = Path(data_dir) / 'train'
    
    for img_dir in sorted(train_dir.iterdir()):
        if not img_dir.is_dir():
            continue
        
        image_file = img_dir / 'image.tif'
        if not image_file.exists():
            continue
        
        # Get image info
        image = tifffile.imread(str(image_file))
        height, width = image.shape[:2]
        image_id = len(images) + 1
        
        images.append({
            'id': image_id,
            'file_name': f"{img_dir.name}/image.tif",
            'height': height,
            'width': width
        })
        
        # Process each class mask
        for class_id in range(1, 5):  # class1 to class4
            mask_file = img_dir / f'class{class_id}.tif'
            if not mask_file.exists():
                continue
            
            mask = tifffile.imread(str(mask_file))
            
            # Get individual instances
            unique_vals = np.unique(mask)
            unique_vals = unique_vals[unique_vals > 0]
            
            for instance_id in unique_vals:
                instance_mask = (mask == instance_id).astype(np.uint8)
                
                # Get bbox
                pos = np.where(instance_mask)
                if len(pos[0]) == 0:
                    continue
                
                y_min, y_max = pos[0].min(), pos[0].max()
                x_min, x_max = pos[1].min(), pos[1].max()
                
                bbox_width = x_max - x_min + 1
                bbox_height = y_max - y_min + 1
                area = instance_mask.sum().item()
                
                # Encode mask to RLE format
                import pycocotools.mask as mask_util
                rle = mask_util.encode(
                    np.asfortranarray(instance_mask)
                )
                
                annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': class_id,
                    'segmentation': {
                        'size': [height, width],
                        'counts': rle['counts'].decode('utf-8')
                    },
                    'area': float(area),
                    'bbox': [float(x_min), float(y_min), 
                            float(bbox_width), float(bbox_height)],
                    'iscrowd': 0
                })
                
                annotation_id += 1
    
    coco_format = {
        'info': {
            'description': 'Medical Image Segmentation Dataset',
            'version': '1.0',
        },
        'images': images,
        'annotations': annotations,
        'categories': [
            {'id': 1, 'name': 'class1'},
            {'id': 2, 'name': 'class2'},
            {'id': 3, 'name': 'class3'},
            {'id': 4, 'name': 'class4'},
        ]
    }
    
    return coco_format


def evaluate_predictions(gt_coco_path, pred_json_path):
    """
    Evaluate predictions using COCO metrics.
    
    Args:
        gt_coco_path: Path to ground truth COCO annotations
        pred_json_path: Path to predictions JSON
    
    Returns:
        dict with metrics
    """
    # Load ground truth
    coco_gt = COCO(gt_coco_path)
    
    # Load predictions
    coco_dt = coco_gt.loadRes(pred_json_path)
    
    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
        'AR1': coco_eval.stats[6],
        'AR10': coco_eval.stats[7],
        'AR100': coco_eval.stats[8],
        'AR_small': coco_eval.stats[9],
        'AR_medium': coco_eval.stats[10],
        'AR_large': coco_eval.stats[11],
    }
    
    return metrics


def main():
    """Main evaluation function."""
    DATA_DIR = './hw3-data-release'
    PRED_FILE = 'test-results.json'
    GT_ANNOTATIONS = 'ground_truth_coco.json'
    
    logger.info("Creating ground truth annotations...")
    
    # Load image mapping
    mapping_file = Path(DATA_DIR) / 'test_image_name_to_ids.json'
    if mapping_file.exists():
        with open(mapping_file) as f:
            image_id_mapping = json.load(f)
    else:
        image_id_mapping = {}
    
    # Create COCO format GT
    coco_gt = create_coco_annotations(DATA_DIR, image_id_mapping)
    
    with open(GT_ANNOTATIONS, 'w') as f:
        json.dump(coco_gt, f)
    
    logger.info(f"Saved ground truth to {GT_ANNOTATIONS}")
    
    if not os.path.exists(PRED_FILE):
        logger.error(f"Predictions file not found: {PRED_FILE}")
        return
    
    logger.info("Evaluating predictions...")
    metrics = evaluate_predictions(GT_ANNOTATIONS, PRED_FILE)
    
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results:")
    logger.info("="*50)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    logger.info("="*50)
    
    # Save metrics
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Metrics saved to evaluation_metrics.json")


if __name__ == '__main__':
    main()
