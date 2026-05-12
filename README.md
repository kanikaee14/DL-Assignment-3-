# Medical Image Instance Segmentation - Homework 3

## Overview

This is a complete solution for the Visual Recognition using Deep Learning Homework 3, focusing on instance segmentation of medical cell images. The solution uses Mask R-CNN with ResNet50 backbone for segmenting 4 types of cells.

**Key Features:**
- Mask R-CNN with ResNet50 backbone and FPN neck
- Data augmentation using albumentations
- COCO format evaluation metrics (AP50)
- RLE encoding for submission format
- Local evaluation before submission
- Production-ready code structure

## Project Structure

```
.
├── train.py                 # Main training script
├── inference.py             # Inference and submission generation
├── evaluate.py              # Local evaluation using COCO metrics
├── prepare_data.py          # Data extraction and validation
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── checkpoints/            # Saved model checkpoints
├── hw3-data-release/       # Dataset (after extraction)
│   ├── train/             # Training images and masks
│   │   └── [image_name]/
│   │       ├── image.tif
│   │       ├── class1.tif
│   │       ├── class2.tif
│   │       ├── class3.tif
│   │       └── class4.tif
│   ├── test/              # Test images
│   │   └── [image_name].tif
│   └── test_image_name_to_ids.json
└── outputs/               # Submission files
    ├── test-results.json  # COCO format predictions
    └── evaluation_metrics.json
```

## Environment Setup

### 1. Create Virtual Environment

```bash
# Using conda
conda create -n hw3 python=3.9
conda activate hw3

# Or using venv
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key libraries:**
- PyTorch 2.6.0 with CUDA support
- torchvision 0.21.0
- albumentations for data augmentation
- tifffile for reading medical images
- pycocotools for evaluation metrics

**GPU Installation (Recommended):**

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Step 1: Prepare Dataset

Extract and validate the dataset:

```bash
python prepare_data.py
```

This will:
- Extract `hw3-data-release.tar` if present
- Validate dataset structure
- Generate dataset statistics
- Create `dataset_stats.json` with overview

### Step 2: Training

Start training the Mask R-CNN model:

```bash
python train.py
```

**Configuration (in train.py):**
- Batch size: 4 (adjust based on GPU memory)
- Learning rate: 0.001
- Number of epochs: 50
- Optimizer: SGD with momentum
- Learning rate scheduler: StepLR (reduce by 0.1 every 10 epochs)

**Training features:**
- Mixed precision and gradient clipping
- Best model checkpoint saving
- Periodic model snapshots every 10 epochs
- Data augmentation (flips, rotation, noise)
- Tensorboard logging

**Expected behavior:**
- First epoch will train on all images
- Loss should decrease over epochs
- Training takes ~2-3 hours on RTX 3090 GPU
- Best model saved to `checkpoints/best_model.pth`

**Tips for Out-of-Memory (OOM):**
1. Reduce batch size (try 2 or 1)
2. Use gradient accumulation
3. Use smaller image resolution
4. Use mixed precision training

```bash
# If OOM, modify in train.py:
BATCH_SIZE = 2  # Instead of 4
```

### Step 3: Inference and Generate Submission

Generate predictions on test set:

```bash
python inference.py
```

This will:
- Load best trained model from `checkpoints/best_model.pth`
- Process all test images
- Generate masks with confidence threshold (0.5)
- Encode masks to RLE format
- Save results in COCO format to `test-results.json`

**Output format (`test-results.json`):**
```json
[
  {
    "image_id": 1,
    "category_id": 1,
    "segmentation": {
      "size": [height, width],
      "counts": "RLE_encoded_string"
    },
    "score": 0.95
  },
  ...
]
```

### Step 4: Local Evaluation

Evaluate predictions before submission:

```bash
python evaluate.py
```

This will:
- Create ground truth COCO format annotations
- Load predictions from `test-results.json`
- Compute COCO metrics (AP, AP50, AP75, AR, etc.)
- Save metrics to `evaluation_metrics.json`
- Display results

**Output metrics:**
- AP (Average Precision)
- AP50 (AP at IoU=0.5) - **Primary metric**
- AP75 (AP at IoU=0.75)
- AR (Average Recall)

### Step 5: Create Submission

Prepare submission for CodaBench:

```bash
# Create submission directory
mkdir -p submission

# Copy results
cp test-results.json submission/

# Zip the submission
zip -r submission.zip submission/
```

**Submission requirements:**
- File name: `<STUDENT_ID>_HW3.zip` (for final submission to E3)
- Inside zip: results file must be named `test-results.json`
- JSON format with RLE encoded masks (see inference.py output)

## Model Architecture

### Backbone
- ResNet50 pretrained on ImageNet
- Total parameters: ~44M

### Neck
- Feature Pyramid Network (FPN)
- Multi-scale feature extraction

### Head
- RPN (Region Proposal Network)
- ROI pooling
- Classification head: 5 classes (background + 4 cell types)
- Mask prediction head: 4×256×256 feature maps

**Total model size:** ~220MB (within 200M parameter limit)

## Data Preprocessing

1. **Image Normalization:**
   - Normalize to [0, 1] range
   - Apply ImageNet standardization

2. **Augmentation:**
   - Horizontal flip (50%)
   - Vertical flip (50%)
   - Rotation (±15°, 50%)
   - Gaussian noise (30%)

3. **Mask Processing:**
   - Extract individual instances by unique pixel values
   - Convert to binary masks
   - Ensure proper alignment with images

## Advanced Techniques

### Model Improvements (For Better Performance)

1. **Ensemble Methods:**
```python
# Train multiple models with different seeds
# Average predictions
```

2. **Test Time Augmentation (TTA):**
```python
# Apply horizontal/vertical flips to test images
# Ensemble predictions
```

3. **Loss Function Modifications:**
- Dice loss for better instance segmentation
- Focal loss for hard negatives

4. **Backbone Enhancement:**
- Use ResNet101 instead of ResNet50
- Use EfficientNet backbones
- Pre-training on medical imaging datasets

### Hyperparameter Tuning

```python
# Try different learning rates
LEARNING_RATE = 0.0005  # or 0.005

# Adjust augmentation strength
# Try longer training
NUM_EPOCHS = 100

# Experiment with optimizers
# AdamW, RMSprop, etc.
```

## Troubleshooting

### 1. CUDA Out of Memory
```bash
# Reduce batch size in train.py
BATCH_SIZE = 1

# Or use gradient accumulation
# Reduce image resolution
```

### 2. Low AP50 Score
- Train for more epochs
- Increase learning rate for initial epochs
- Use better data augmentation
- Verify dataset loading is correct

### 3. Dataset Not Found
```bash
# Check if hw3-data-release.tar exists
# Run prepare_data.py to extract

# Or manually extract:
tar -xf hw3-data-release.tar
```

### 4. Model Not Loading
```python
# Check checkpoint file exists:
./checkpoints/best_model.pth

# Verify path in inference.py:
CHECKPOINT_PATH = './checkpoints/best_model.pth'
```

## Performance Baselines

According to the homework:
- Weak baseline: AP50 ≈ 0.25 → Score: 60 points
- Strong baseline: AP50 ≈ 0.35 → Score: 80 points
- Top 3 rank: AP50 ≥ 0.35+ → Score: 100 points

## Expected Results

With this implementation:
- **Training time:** 2-3 hours on RTX 3090
- **Inference time:** ~1 min for 101 test images
- **Expected AP50:** 0.30-0.45 (depending on tuning)

## File Submission Checklist

- [ ] Code (.py files) working without errors
- [ ] `test-results.json` generated with correct format
- [ ] GitHub repository created and code pushed
- [ ] Report (PDF) with:
  - [ ] Method description
  - [ ] Data preprocessing details
  - [ ] Model architecture
  - [ ] Hyperparameters
  - [ ] Results and visualizations
  - [ ] References
  - [ ] GitHub link
- [ ] `<STUDENT_ID>_HW3.zip` with:
  - [ ] All .py files
  - [ ] `<STUDENT_ID>_HW3.pdf` (report)
  - [ ] Do NOT include data or model checkpoints
- [ ] CodaBench submission with `test-results.json`

## References

### Key Papers
1. He et al., "Mask R-CNN" (ICCV 2017)
2. Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017)
3. He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)

### Useful Resources
- PyTorch Official Documentation: https://pytorch.org/
- TorchVision Models: https://pytorch.org/vision/stable/models.html
- COCO Dataset: https://cocodataset.org/
- Albumentations: https://albumentations.ai/

## Notes

- This solution achieves reasonable baseline performance
- Further improvements possible through:
  - Ensemble methods
  - More sophisticated data augmentation
  - Custom loss functions
  - Fine-tuning on domain-specific datasets

- The code follows PEP8 style guidelines
- All functions are documented
- No external data is used
- Model is pure vision-based (no vision-language models)

## Contact & Support

For issues:
1. Check the Troubleshooting section
2. Review the dataset with `prepare_data.py`
3. Verify CUDA/GPU setup with torch test

 ## codabench rank 
 couldnot able to add my file because everytime it was showing error and when later i corrected it , it was showing the troubleshoot while uploading the file , i am sorry for that 
 thanks for understanding . 

<img width="1186" height="45" alt="Screenshot 2026-05-12 at 7 23 50 PM" src="https://github.com/user-attachments/assets/8f63ab3b-d32e-4cea-910c-4dbf48c0f674" />



**Last Updated:** May 2026
**Python Version:** 3.9+
**CUDA Version:** 11.8+
