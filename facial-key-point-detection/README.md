# Facial Keypoint Detection

A PyTorch-based implementation of a facial landmark regressor, developed as part of an assignment for the Graphics & Media Lab (Vision Group).

The task is to train a custom Convolutional Neural Network (CNN) to predict the coordinates of 14 keypoints (28 coordinates in total: $[x_1, y_1, \dots, x_{14}, y_{14}]$) on human faces.

---

## Constraints

- **Pure PyTorch** — No high-level wrappers (e.g. PyTorch Lightning) or pre-built model architectures are permitted. The network architecture, training loop, and dataset pipeline are built from scratch.
- **Self-contained** — No external training data may be used.
- **Input Images** — Images can be grayscale or RGB. The pipeline automatically converts grayscale inputs into 3-channel images.
- **Metric** — Quality is evaluated using Mean Squared Error (MSE) normalized to a $100 \times 100$ image scale:

$$Q(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (f(x_i) - y_i)^2$$

- **Resource Limits** — Strict runtime and memory constraints apply during evaluation.

---

## Architecture

### Custom CNN (`facepointsModel`)

The network consists of a sequential feature extractor followed by a regression head.

**Feature Extractor**

```
Conv2D(3, 64, kernel=3)    -> ReLU -> MaxPool2D(2, 2)
Conv2D(64, 128, kernel=3)  -> ReLU -> MaxPool2D(2, 2)
Conv2D(128, 128, kernel=3) -> ReLU -> MaxPool2D(2, 2)
Conv2D(128, 256, kernel=3) -> ReLU
```

**Regressor Head**

```
Flatten -> Linear(256 * 28 * 28, 64) -> ReLU -> Linear(64, 28)
```

---

## Data Pipeline & Augmentation

- **Normalization** — Prior to training, the pipeline computes pixel-wise mean and standard deviation across the entire training set to normalize inputs accurately.
- **Augmentations** (via [Albumentations](https://albumentations.ai/)):
  - Adaptive resizing to $224 \times 224$ pixels.
  - Spatial: `ShiftScaleRotate` up to 30 degrees, with landmark coordinates updated automatically.
  - Color: `RandomBrightnessContrast` for robustness across varying lighting conditions.

---

## Installation

```bash
pip install torch albumentations opencv-python pillow pandas numpy
```

---

## API

### Training — `train_detector`

Trains the CNN using a provided annotations dictionary and image folder.

```python
from detection import train_detector

# Annotations: maps each image filename to a flat list of 28 floats [x1, y1, ..., x14, y14]
annotations = {
    "image_01.jpg": [x1, y1, x2, y2, ..., x14, y14],
    # ...
}

model = train_detector(
    dictionary=annotations,
    root="path/to/images_folder",
    fast_train=False
)
```

**Fast Train Mode** (`fast_train=True`)

For local validation and grading-system checks, this mode:

- Runs on CPU only.
- Limits training to 1 epoch with a reduced batch size.
- Disables parallel data-loading and disk writes (compatible with read-only environments).

---

### Inference — `detect`

Runs predictions on a directory of images using a saved model checkpoint (`facepoints_model.pt`). Keypoint coordinates are automatically rescaled to each image's original dimensions.

```python
from detection import detect

results = detect(
    model_path="facepoints_model.pt",
    images_folder="path/to/test_images"
)

# Returns: { "filename.jpg": [x1, y1, ..., x14, y14] }
print(results)
```

---

## Local Evaluation

To validate your implementation against the public dataset locally:

```bash
python run.py path/to/public_tests
```

The script holds out a validation split from the public training set. Early stopping is triggered if validation loss does not improve for 6 consecutive epochs.
