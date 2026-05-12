import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tifffile
import logging
from pathlib import Path
import gc
from torchvision import transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment and backend setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ============================================================================
# IMPROVED MEDICAL IMAGE PREPROCESSING
# ============================================================================
def normalize_medical_image(image):
    """Better normalization for medical images"""
    image = image.astype(np.float32)
    
    # Remove extreme outliers more carefully
    p1, p99 = np.percentile(image, (1, 99))
    image = np.clip(image, p1, p99)
    
    # Normalize to [0, 1]
    image_min = image.min()
    image_max = image.max()
    
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)
    
    image = np.clip(image, 0.0, 1.0)
    return image


# ============================================================================
# DATA AUGMENTATION FOR MEDICAL IMAGES
# ============================================================================
class MedicalImageAugmentation:
    """Augmentation specifically for medical images"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, masks):
        # Random horizontal flip
        if np.random.random() < self.p:
            image = np.fliplr(image)
            masks = [np.fliplr(m) for m in masks]
        
        # Random vertical flip
        if np.random.random() < self.p:
            image = np.flipud(image)
            masks = [np.flipud(m) for m in masks]
        
        # Random rotation (90, 180, 270 degrees for medical images)
        if np.random.random() < self.p:
            k = np.random.choice([1, 2, 3])  # 90, 180, 270
            image = np.rot90(image, k)
            masks = [np.rot90(m, k) for m in masks]
        
        # Random brightness adjustment
        if np.random.random() < self.p:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() < self.p:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean_val = image.mean()
            image = mean_val + (image - mean_val) * contrast_factor
            image = np.clip(image, 0, 1)
        
        return image, masks


# ============================================================================
# IMPROVED DATASET CLASS
# ============================================================================
class MedicalSegmentationDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.root_dir = Path(root_dir)
        self.image_paths = sorted((self.root_dir / 'train').glob('*/image.tif'))
        self.augment = augment
        self.augmentor = MedicalImageAugmentation(p=0.5) if augment else None
        
        # Filter out images with NO objects at all
        self.valid_indices = []
        for idx, image_path in enumerate(self.image_paths):
            image_dir = image_path.parent
            has_any = False
            for class_id in range(1, 5):
                mask_path = image_dir / f'class{class_id}.tif'
                if mask_path.exists():
                    has_any = True
                    break
            if has_any:
                self.valid_indices.append(idx)
        
        logger.info(f"Total images: {len(self.image_paths)}, Valid (with objects): {len(self.valid_indices)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        image_path = self.image_paths[actual_idx]
        image_dir = image_path.parent
        
        # Load and normalize image
        image = tifffile.imread(str(image_path))
        image = normalize_medical_image(image)
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=2)
        
        # Load masks
        masks = []
        labels = []
        h, w = image.shape[:2]
        
        for class_id in range(1, 5):
            mask_path = image_dir / f'class{class_id}.tif'
            if mask_path.exists():
                mask = tifffile.imread(str(mask_path))
                unique_vals = np.unique(mask)
                unique_vals = unique_vals[unique_vals > 0]
                
                for val in unique_vals:
                    instance = (mask == val).astype(np.uint8)
                    # Filter out tiny objects (noise)
                    if instance.sum() > 50:  # Increased from 100 to 50 but still filter
                        masks.append(instance)
                        labels.append(class_id)
        
        # ❌ REMOVED THE FAKE DATA GENERATION ❌
        # NO MORE DUMMY MASKS!
        
        # If no objects found, skip this image (handled by filtering in __init__)
        if len(masks) == 0:
            # This shouldn't happen now, but just in case
            masks = [np.zeros((h, w), dtype=np.uint8)]
            labels = [1]
        
        # Limit masks to avoid memory issues (but keep all valid ones)
        if len(masks) > 10:
            indices = np.random.choice(len(masks), 10, replace=False)
            masks = [masks[i] for i in sorted(indices)]
            labels = [labels[i] for i in sorted(indices)]
        
        # Apply augmentation
        if self.augment:
            image, masks = self.augmentor(image, masks)
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        masks = torch.stack([torch.from_numpy(m) for m in masks])
        
        # Generate bounding boxes
        boxes = []
        valid_masks = []
        valid_labels = []
        
        for i, mask in enumerate(masks):
            rows, cols = torch.where(mask > 0)
            if len(rows) > 0:
                rmin, rmax = rows.min().item(), rows.max().item()
                cmin, cmax = cols.min().item(), cols.max().item()
                
                # Add small padding to boxes
                pad = 2
                x1 = max(0, cmin - pad)
                y1 = max(0, rmin - pad)
                x2 = min(w - 1, cmax + pad)
                y2 = min(h - 1, rmax + pad)
                
                # Ensure valid box
                if x2 > x1 and y2 > y1:
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    valid_masks.append(mask)
                    valid_labels.append(labels[i])
        
        if len(boxes) == 0:
            # Shouldn't happen, but add minimal box if it does
            boxes = [[10.0, 10.0, 20.0, 20.0]]
            valid_masks = [masks[0]]
            valid_labels = [labels[0]]
        
        valid_masks = torch.stack(valid_masks)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(valid_labels, dtype=torch.int64),
            'masks': valid_masks,
            'image_id': torch.tensor([actual_idx]),
            'area': torch.as_tensor([m.sum().item() for m in valid_masks]).float(),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return image, target


# ============================================================================
# MODEL SETUP
# ============================================================================
def get_model(num_classes=5):
    """Get Mask R-CNN model with pretrained backbone"""
    model = maskrcnn_resnet50_fpn(weights='COCO_V1')
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    # ============ OPTIONAL: Tune RPN for medical images ============
    # Medical images might have different object sizes
    # You can adjust anchors if needed:
    # model.rpn.anchor_generator.sizes = ((32, 64, 128, 256, 512),)
    # model.rpn.anchor_generator.aspect_ratios = ((0.5, 1.0, 2.0),)
    
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, scaler, epoch, device, num_epochs):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets_device = []
        
        for target in targets:
            target_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in target.items()
            }
            targets_device.append(target_device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # NaN protection
        for img in images:
            torch.nan_to_num_(img, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda', enabled=True):
            losses = model(images, targets_device)
            loss = sum(l for l in losses.values())
            
            # Skip bad batches
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping...")
                del images, targets_device, losses, loss
                torch.cuda.empty_cache()
                continue
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        valid_batches += 1
        
        # Cleanup
        del images, targets_device, losses, loss
        torch.cuda.empty_cache()
        
        if batch_idx % 5 == 0:
            mem = torch.cuda.memory_allocated() / 1e9
            avg_loss = total_loss / max(1, valid_batches)
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, "
                f"Loss: {avg_loss:.4f}, Mem: {mem:.1f}GB"
            )
    
    return total_loss / max(1, valid_batches)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    device = torch.device('cuda')
    DATA_DIR = './hw3-data-release'
    BATCH_SIZE = 4  # Increased from 1 to 4
    EPOCHS = 40  # Increased from 30
    LEARNING_RATE = 0.01  # Increased from 0.005
    OUTPUT_DIR = './checkpoints'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"🚀 IMPROVED TRAINING - {torch.cuda.get_device_name(0)}")
    
    # Load dataset with augmentation
    dataset = MedicalSegmentationDataset(DATA_DIR, augment=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,  # Increased from 0
        collate_fn=collate_fn, 
        pin_memory=True  # Changed from False to True
    )
    
    logger.info(f"Dataset: {len(dataset)} valid images")
    
    model = get_model(num_classes=5)
    model.to(device)
    
    # Better optimizer setup
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=0.9, 
        weight_decay=1e-4, 
        nesterov=True
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Better learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        gc.collect()
        
        epoch_loss = train_one_epoch(
            model, dataloader, optimizer, scaler, epoch, device, EPOCHS
        )
        
        # Save checkpoint if improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            logger.info(f"✅ New best model saved with loss: {epoch_loss:.4f}")
        
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} completed. "
            f"Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}"
        )


if __name__ == '__main__':
    main()
