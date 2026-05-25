# Interactive Image Segmentation with Click Guidance

A PyTorch implementation of an interactive image segmentation model based on the [RITM](https://github.com/saic-vul/ritm_interactive_segmentation) (Reviving Iterative Training with Mask Guidance) framework. The model uses a modified DeepLabv3 with a MobileNetV3-Large backbone to perform iterative object selection guided by user clicks.

Users refine mask predictions by placing:
- **Positive clicks** — add regions to the predicted mask
- **Negative clicks** — remove background regions from the predicted mask

---

## How It Works

Each interaction round accepts an image and the current set of user clicks, producing a refined segmentation mask. Mask quality (measured by IoU) improves with each additional click.

```
[User Clicks] ──┐
                v
[RGB Image] ──► [6-Ch Input] ──► [DeepLabv3 + MobileNetV3] ──► [Refined Mask]
                ^                                                       │
                └──────────────── [Next round feedback] ────────────────┘
```

---

## Architecture

### 6-Channel Input

The first convolutional layer of the DeepLabv3 backbone is modified to accept 6 input channels:

| Channels | Content |
|---|---|
| 3 | RGB image |
| 1 | Predicted mask from the previous round |
| 1 | Positive click disk map |
| 1 | Negative click disk map |

### Iterative Training (`ISTrainer`)

- For each training sample, a random number of clicks are simulated inside and outside the ground truth mask.
- Predictions are fed back into the model each round, simulating real user interaction.
- Each new click is placed at the center of the largest error region between the predicted mask and ground truth (via distance transform).

### Loss Functions

- `BCEWithLogitsLoss` — standard binary cross-entropy baseline.
- `NormalizedFocalLossSigmoid` — adapted from the official RITM repository to address class imbalance in interactive masks.

### Augmentations

Custom spatial augmentations (e.g. `UniformRandomResize`) correctly rescale and translate user-click coordinates alongside the image.

---

## Repository Structure

```
├── solution.py              # Model, predictor, and trainer implementation
├── run.py                   # Evaluation script
├── COCO_LVIS/               # Training dataset directory
├── tests/
│   └── 00_test_davis_input/ # Public DAVIS test inputs
└── utils/
    ├── datasets.py          # COCO/LVIS dataset loaders
    ├── points_sampler.py    # Click sampler for training
    ├── clicker.py           # Click simulation helper
    └── misc.py              # Visualization and saving utilities
```

---

## Installation

```bash
pip install torch torchvision albumentations opencv-python tqdm numpy
```

---

## Dataset Preparation

The model trains on a merged subset of COCO and LVIS. Run the following to download and extract the required files:

```bash
mkdir -p COCO_LVIS/

# Annotations
wget -N https://courses.cv-gml.ru/storage/tasks/interactive-segmentation/cocolvis_annotation.tar.gz
tar -xf cocolvis_annotation.tar.gz -C COCO_LVIS/

# COCO 2017 train images
wget -N http://images.cocodataset.org/zips/train2017.zip
unzip -jqo train2017.zip -d COCO_LVIS/train/images/

# COCO 2017 val images
wget -N http://images.cocodataset.org/zips/val2017.zip
unzip -jqo val2017.zip -d COCO_LVIS/val/images/
```

---

## Training

```bash
python solution.py
```

The trainer automatically loads pretrained MobileNetV3 ImageNet weights, configures a StepLR scheduler, and runs the full training loop over the COCO/LVIS dataset.

---

## Evaluation

The model is evaluated using **IoU@3** — Intersection over Union after exactly 3 interactive clicks.

```bash
python run.py tests
```

This will:
1. Load `checkpoint.pt`
2. Simulate sequential clicks targeting the largest error region
3. Write visual results to `tests/00_test_davis_check/output/`

### Scoring

| IoU@3 | Score |
|---|---|
| $\geq 0.81$ | 10 pts |
| $\geq 0.80$ | 9 pts |
| $\geq 0.79$ | 8 pts |
| $\geq 0.78$ | 7 pts |
| $\geq 0.77$ | 6 pts |
| $\geq 0.76$ | 5 pts |
| $\geq 0.75$ | 4 pts |
| $\geq 0.74$ | 3 pts |
| $\geq 0.72$ | 2 pts |
| $\geq 0.70$ | 1 pt |

---

## Potential Improvements

To push IoU above the 0.81 threshold:

- **Test-Time Augmentations (TTA)** on click inputs
- **Stronger backbones** — ViT (SimpleClick) or SAM adapter
- **Higher resolution training** — scale canvas from 400×400 upward
- **Click-region loss weighting** — apply higher loss weight in regions surrounding interaction clicks
