import json
import torch
import numpy as np
import tifffile
from pathlib import Path
import logging

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import pycocotools.mask as mask_util

# =========================
# CONFIG
# =========================
BASE_DIR = Path("hw3-data-release")
TEST_DIR = BASE_DIR / "test_release"
MAPPING_PATH = BASE_DIR / "test_image_name_to_ids.json"

OUTPUT_PATH = "test-results.json"
CHECKPOINT_PATH = "./checkpoints/best_model.pth"

# Thresholds
CONF_THRESHOLD = 0.05
MASK_THRESHOLD = 0.3

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# IMAGE NORMALIZATION
# =========================
def normalize_medical_image(image):
    """Better normalization for medical images"""
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, (1, 99))
    image = np.clip(image, p1, p99)
    image_min = image.min()
    image_max = image.max()
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)
    image = np.clip(image, 0.0, 1.0)
    return image


# =========================
# LOAD filename -> id mapping
# =========================
def load_mapping():
    with open(MAPPING_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {x["file_name"]: x["id"] for x in data}
    return data


# =========================
# Load model
# =========================
def load_model():
    model = maskrcnn_resnet50_fpn(weights=None)

    num_classes = 5  # background + 4 classes

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    if Path(CHECKPOINT_PATH).exists():
        logger.info(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model.eval()
    return model


# =========================
# Encode mask to RLE
# =========================
def encode_rle(mask):
    """Encode binary mask to RLE format"""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


# =========================
# Run inference on single image
# =========================
def predict(model, image, device):
    """Run inference on a single image"""
    image = normalize_medical_image(image)

    # Fix channels
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

    tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    return output


# =========================
# Post-process output
# =========================
def postprocess_output(output, conf_threshold=0.5, mask_threshold=0.5):
    """Filter predictions based on confidence and mask thresholds"""
    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    masks = output["masks"].cpu().numpy()

    results = []

    for i in range(len(scores)):
        if scores[i] < conf_threshold:
            continue
        mask = masks[i, 0] > mask_threshold
        if mask.sum() == 0:
            continue
        x1, y1, x2, y2 = boxes[i]
        results.append({
            'score': float(scores[i]),
            'label': int(labels[i]),
            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            'mask': mask,
            'rle': encode_rle(mask)
        })

    return results


# =========================
# Main function
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Check folders
    assert TEST_DIR.exists(), "❌ test_release folder missing"
    assert MAPPING_PATH.exists(), "❌ mapping file missing"

    # Load filename to id mapping
    filename_to_id = load_mapping()

    # Debug: Print some mappings
    print("Sample filename to id mappings:")
    for i, (fname, id_val) in enumerate(filename_to_id.items()):
        print(f"{fname} -> {id_val}")
        if i >= 10:
            break

    # Load model
    model = load_model().to(device)

    # List test images
    images = sorted(list(TEST_DIR.glob("*.tif")))
    logger.info(f"Found {len(images)} test images")

    results = []
    failed_images = []

    for idx, img_path in enumerate(images):
        filename = img_path.name
        image_id = filename_to_id.get(filename)

        # Debug: Check filename and id
        if image_id is None:
            print(f"WARNING: {filename} not found in mapping!")
            failed_images.append(filename)
            continue
        else:
            print(f"Processing {filename} with ID {image_id}")

        try:
            # Load image
            image = tifffile.imread(str(img_path))
            # Run inference
            output = predict(model, image, device)
            # Post-process
            predictions = postprocess_output(
                output,
                conf_threshold=CONF_THRESHOLD,
                mask_threshold=MASK_THRESHOLD
            )
            # Save results
            for pred in predictions:
                results.append({
                    "image_id": int(image_id),
                    "category_id": pred['label'],
                    "score": pred['score'],
                    "bbox": pred['bbox'],
                    "segmentation": {
                        "size": list(pred['mask'].shape),
                        "counts": pred['rle']
                    }
                })

            if idx % 10 == 0:
                print(f"Processed {idx + 1}/{len(images)} images, found {len(predictions)} predictions in image {image_id}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            failed_images.append(filename)
            continue

    # Save JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("=" * 60)
    print("✅ INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total images: {len(images)}")
    print(f"Successfully processed: {len(images) - len(failed_images)}")
    print(f"Failed: {len(failed_images)}")
    print(f"Total predictions: {len(results)}")
    print(f"Avg predictions per image: {len(results) / max(1, (len(images) - len(failed_images))):.2f}")
    print(f"Output saved to: {OUTPUT_PATH}")

    if failed_images:
        print(f"Failed images: {failed_images}")

if __name__ == "__main__":
    main()