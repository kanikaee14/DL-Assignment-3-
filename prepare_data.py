"""
Data preparation and validation script.
Extracts tar file and validates dataset structure.
"""

import os
import tarfile
import json
import numpy as np
import tifffile
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_tar_file(tar_path, extract_path='.'):
    """Extract tar file."""
    logger.info(f"Extracting {tar_path} to {extract_path}")
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_path)
    logger.info("Extraction completed!")


def validate_dataset(data_dir):
    """
    Validate dataset structure and content.
    
    Args:
        data_dir: Path to dataset directory
    """
    data_dir = Path(data_dir)
    
    logger.info("\n" + "="*50)
    logger.info("Dataset Validation")
    logger.info("="*50)
    
    # Check train directory
    train_dir = data_dir / 'train'
    if not train_dir.exists():
        logger.error("Train directory not found!")
        return False
    
    train_images = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(train_images)} training images")
    
    # Validate train images
    for img_dir in train_images[:3]:  # Check first 3
        image_file = img_dir / 'image.tif'
        if image_file.exists():
            try:
                image = tifffile.imread(str(image_file))
                logger.info(
                    f"  {img_dir.name}: image shape={image.shape}, "
                    f"dtype={image.dtype}, range=[{image.min()}, {image.max()}]"
                )
                
                # Check masks
                mask_files = sorted(img_dir.glob('class*.tif'))
                for mask_file in mask_files:
                    mask = tifffile.imread(str(mask_file))
                    unique_vals = np.unique(mask)
                    logger.info(
                        f"    {mask_file.name}: shape={mask.shape}, "
                        f"unique_values={len(unique_vals)}, "
                        f"range=[{unique_vals.min()}, {unique_vals.max()}]"
                    )
            except Exception as e:
                logger.error(f"Error reading {img_dir.name}: {e}")
    
    # Check test directory
    test_dir = data_dir / 'test'
    if test_dir.exists():
        test_images = sorted(test_dir.glob('*.tif'))
        logger.info(f"\nFound {len(test_images)} test images")
        
        for test_file in test_images[:3]:  # Check first 3
            try:
                image = tifffile.imread(str(test_file))
                logger.info(
                    f"  {test_file.name}: shape={image.shape}, "
                    f"dtype={image.dtype}"
                )
            except Exception as e:
                logger.error(f"Error reading {test_file.name}: {e}")
    
    # Check image_id mapping
    mapping_file = data_dir / 'test_image_name_to_ids.json'
    if mapping_file.exists():
        with open(mapping_file) as f:
            mapping = json.load(f)
        logger.info(f"\nImage ID mapping found with {len(mapping)} entries")
        for i, (filename, img_info) in enumerate(list(mapping.items())[:2]):
            logger.info(f"  {filename}: id={img_info.get('id', 'N/A')}")
    
    logger.info("\n" + "="*50)
    logger.info("Validation completed!")
    logger.info("="*50)
    
    return True


def generate_dataset_statistics(data_dir):
    """Generate dataset statistics."""
    data_dir = Path(data_dir)
    
    logger.info("\n" + "="*50)
    logger.info("Dataset Statistics")
    logger.info("="*50)
    
    train_dir = data_dir / 'train'
    stats = {
        'total_images': 0,
        'total_instances': 0,
        'instances_per_class': defaultdict(int),
        'image_sizes': [],
        'mask_stats': defaultdict(list)
    }
    
    for img_dir in train_dir.iterdir():
        if not img_dir.is_dir():
            continue
        
        stats['total_images'] += 1
        image_file = img_dir / 'image.tif'
        
        if image_file.exists():
            image = tifffile.imread(str(image_file))
            stats['image_sizes'].append(image.shape)
            
            # Analyze masks
            for class_id in range(1, 5):
                mask_file = img_dir / f'class{class_id}.tif'
                if mask_file.exists():
                    mask = tifffile.imread(str(mask_file))
                    unique_vals = np.unique(mask)
                    num_instances = len(unique_vals[unique_vals > 0])
                    
                    stats['total_instances'] += num_instances
                    stats['instances_per_class'][f'class{class_id}'] += num_instances
                    
                    if num_instances > 0:
                        area_stats = []
                        for val in unique_vals[unique_vals > 0]:
                            area = (mask == val).sum()
                            area_stats.append(area)
                        stats['mask_stats'][f'class{class_id}'].extend(area_stats)
    
    # Compute statistics
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Total instances: {stats['total_instances']}")
    logger.info(f"Average instances per image: {stats['total_instances']/max(stats['total_images'], 1):.2f}")
    
    logger.info("\nInstances per class:")
    for class_name, count in sorted(stats['instances_per_class'].items()):
        logger.info(f"  {class_name}: {count}")
    
    if stats['image_sizes']:
        image_sizes = np.array(stats['image_sizes'])
        logger.info(f"\nImage size statistics:")
        logger.info(f"  Height: min={image_sizes[:, 0].min()}, "
                   f"max={image_sizes[:, 0].max()}, "
                   f"mean={image_sizes[:, 0].mean():.1f}")
        logger.info(f"  Width: min={image_sizes[:, 1].min()}, "
                   f"max={image_sizes[:, 1].max()}, "
                   f"mean={image_sizes[:, 1].mean():.1f}")
    
    logger.info("\nMask area statistics (pixels):")
    for class_name in sorted(stats['mask_stats'].keys()):
        areas = stats['mask_stats'][class_name]
        if areas:
            areas = np.array(areas)
            logger.info(f"  {class_name}: min={areas.min()}, "
                       f"max={areas.max()}, mean={areas.mean():.1f}")
    
    logger.info("="*50 + "\n")
    
    return stats


def main():
    """Main data preparation function."""
    TAR_FILE = 'hw3-data-release.tar'
    DATA_DIR = './hw3-data-release'
    
    # Extract tar file
    if os.path.exists(TAR_FILE) and not os.path.exists(DATA_DIR):
        logger.info(f"Found {TAR_FILE}, extracting...")
        extract_tar_file(TAR_FILE)
    elif not os.path.exists(DATA_DIR):
        logger.warning(f"Neither {TAR_FILE} nor {DATA_DIR} found!")
        return
    else:
        logger.info(f"Dataset already extracted at {DATA_DIR}")
    
    # Validate dataset
    if validate_dataset(DATA_DIR):
        # Generate statistics
        stats = generate_dataset_statistics(DATA_DIR)
        
        # Save statistics
        stats_file = os.path.join(DATA_DIR, 'dataset_stats.json')
        with open(stats_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            stats_dict = {
                'total_images': stats['total_images'],
                'total_instances': stats['total_instances'],
                'instances_per_class': dict(stats['instances_per_class']),
            }
            json.dump(stats_dict, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")


if __name__ == '__main__':
    main()
