# Fine-Tuning MobileNetV2 for Bird Species Classification

An end-to-end image classification pipeline for identifying 50 bird species, built with PyTorch, PyTorch Lightning, and Albumentations. The project implements a transfer learning workflow that adapts a pre-trained MobileNetV2 backbone to achieve high accuracy under tight hardware and memory constraints.

Developed as part of an assignment for the Graphics & Media Lab (Vision Group).

---

## Overview

Deep neural networks trained from scratch on small datasets are highly susceptible to overfitting. This project addresses that by implementing fine-tuned transfer learning — reusing feature representations learned on ImageNet and adapting them to a target dataset of bird images.

Key design decisions:

- **Stratified Validation Split** — Ensures equal class representation across all 50 bird species in both the training (80%) and validation (20%) splits.
- **Regularization** — Combines spatial and pixel-level augmentations, dual-stage Dropout, Batch Normalization, and weight decay to combat overfitting.
- **Modular Pipeline** — PyTorch Lightning separates model logic from data handling for clean, reproducible code.
- **Production Constraints** — Designed for sandboxed runtimes: no internet access, strict time limits, CPU-only inference.

---

## Architecture

Built on a pre-trained MobileNetV2 backbone optimized for resource-constrained environments.

```
Input Image (224 x 224 x 3)
        |
        v
[MobileNetV2 Feature Extractor]  ->  Early layers frozen (universal features preserved)
        |
        v
[Unfrozen Feature Layers]        ->  Last 6 residual blocks fine-tuned
        |
        v
[Adaptive Average Pooling]       ->  1D spatial representation
        |
        v
[Custom Classification Head]
  |- BatchNorm1d
  |- Dropout(0.3)
  |- Linear(1280 -> 300)
  |- BatchNorm1d
  |- Dropout(0.5)
  |- ReLU
  |- Linear(300 -> 50 classes)
```

**Fine-Tuning Strategy**

To prevent catastrophic forgetting of generic low-level features (edges, textures) while adapting to fine-grained bird attributes:

- Early and intermediate backbone layers are frozen (`requires_grad = False`).
- The last 6 convolutional/inverted residual blocks are unfrozen to learn high-level semantic features.
- A regularized MLP head with batch normalization and dual dropout is stacked on the global average pooling output.

---

## Data Augmentation

Training augmentations are applied via [Albumentations](https://albumentations.ai/):

| Category | Transform | Details |
|---|---|---|
| Spatial | `HorizontalFlip` | $p = 0.9$ |
| Spatial | `ShiftScaleRotate` | Rotation up to 45°, $p = 0.5$ |
| Spatial | `Perspective` | Scale $(0.05, 0.1)$, $p = 0.5$ |
| Color | `RandomBrightnessContrast` | $p = 0.5$ |
| Noise | `GaussNoise` | $p = 1.0$ |
| Normalization | `Normalize` | ImageNet mean/std |

Validation uses only resize and normalization.

---

## Project Structure

All core logic lives in `classification.py`:

| Component | Description |
|---|---|
| `BirdDataset` | Custom `Dataset` — handles image loading, RGB conversion, and lazy augmentation |
| `BirdDataModule` | `LightningDataModule` — manages stratified splitting, transforms, and `DataLoader` creation |
| `BirdClassifier` | `LightningModule` — defines the network, optimizer, LR scheduling, and training/validation steps |
| `train_classifier` | Top-level trainer — configures seed, device (CUDA/MPS/CPU), and checkpoint saving |
| `classify` | Inference pipeline — loads a saved checkpoint and returns per-image predictions |

---

## Installation

```bash
pip install torch torchvision pytorch-lightning albumentations scikit-learn numpy pillow opencv-python
```

---

## Usage

### Training — `train_classifier`

```python
from classification import train_classifier

train_gt = {
    "bird_01.jpg": 12,
    "bird_02.jpg": 45,
    # filename: class_index
}

model = train_classifier(
    train_gt=train_gt,
    train_img_dir="path/to/train_images/",
    fast_train=False  # True for a 1-epoch sanity check
)
```

**Fast Train Mode** (`fast_train=True`)

- Runs on CPU only with a reduced batch size.
- Limits training to 1 epoch.
- Skips checkpoint writing — compatible with read-only evaluation environments.

### Inference — `classify`

```python
from classification import classify

predictions = classify(
    model_path="birds_model.pt",
    test_img_dir="path/to/test_images/"
)

# Returns: { "test_img_01.jpg": 12, ... }
print(predictions)
```

---

## Evaluation

Model performance is mapped to the following accuracy thresholds:

| Accuracy | Score |
|---|---|
| $\geq 0.88$ | 10 / 10 |
| $\geq 0.86$ | 9 / 10 |
| $\geq 0.83$ | 8 / 10 |
| $\geq 0.80$ | 7 / 10 |
| $\geq 0.75$ | 6 / 10 |
| $\geq 0.70$ | 5 / 10 |
| $\geq 0.60$ | 3 / 10 |
