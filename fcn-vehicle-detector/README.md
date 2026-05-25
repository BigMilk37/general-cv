# Fully Convolutional Vehicle Detector

A PyTorch-based vehicle detection pipeline that trains a binary CNN classifier, converts it into a Fully Convolutional Network (FCN) for efficient sliding-window inference, and evaluates performance using custom IoU, Precision-Recall, and Non-Maximum Suppression implementations.

---

## Overview

Naive sliding-window detection — repeatedly cropping and classifying image regions — is computationally expensive at scale. This project addresses that by:

1. Training a CNN classifier on a fixed $40 \times 100$ receptive field.
2. Converting the trained classifier's dense layers into equivalent convolutional layers, producing an FCN that processes full images in a single forward pass.
3. Generating a spatial confidence heatmap over the input image.
4. Applying Non-Maximum Suppression (NMS) to filter duplicate detections.
5. Evaluating detection quality via a Precision-Recall curve and AUC metric.

---

## Pipeline

### 1. Binary Classifier

A CNN trained to distinguish vehicle patches from background regions.

- **Input:** $(1, 40, 100)$ — single-channel grayscale image patch
- **Output:** 2 logits — `[Background, Vehicle]`
- **Architecture:** Four convolutional blocks (Conv2D → BatchNorm → Dropout → ReLU → MaxPool), followed by a linear classification head

### 2. FCN Conversion

To enable efficient inference on images of arbitrary resolution, the `Flatten` + `Linear` layers are replaced with equivalent `Conv2d` layers — preserving all learned weights and producing identical outputs on $40 \times 100$ patches.

Given a feature map of shape $C_{in} \times H \times W$ entering a Flatten layer followed by a Linear layer with $C_{out}$ output units, the substitution is:

- A single `Conv2d` with output channels $C_{out}$ and kernel size $H \times W$
- Subsequent `Linear` layers become `Conv2d` layers with $1 \times 1$ kernels

$$[C_{out},\ C_{in} \cdot H \cdot W] \implies [C_{out},\ C_{in},\ H,\ W]$$

### 3. Detection Retrieval

Passing a full image through the FCN produces a $2 \times H_{heat} \times W_{heat}$ confidence grid. With 4 pooling layers, the effective spatial stride is 16 pixels. Each output position $(i, j)$ maps to:

$$y = i \times 16, \quad x = j \times 16, \quad h = 40, \quad w = 100$$

---

## Evaluation Metrics

### Intersection over Union (IoU)

$$IoU(B_{pred}, B_{gt}) = \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}$$

A prediction with $IoU \geq 0.5$ against a ground truth box is a True Positive candidate.

### Precision-Recall Curve & AUC

Predictions are sorted by descending confidence and matched greedily to ground truth boxes. Cumulative Precision and Recall are computed at each threshold:

$$\text{Precision} = \frac{TP_{cum}}{TP_{cum} + FP_{cum}}, \qquad \text{Recall} = \frac{TP_{cum}}{\text{Total Ground Truth}}$$

AUC is computed via the trapezoidal rule:

$$AUC = \sum_{i=1}^{N} (R_i - R_{i-1}) \cdot \frac{P_i + P_{i-1}}{2}$$

### Non-Maximum Suppression (NMS)

Overlapping detections from adjacent sliding-window positions are filtered as follows:

1. Sort all predictions by confidence (descending).
2. Keep the highest-scoring box.
3. Suppress all subsequent boxes with $IoU > 0.37$ against the kept box.
4. Repeat for remaining boxes.

NMS improves final validation AUC from approximately 20% to $\geq 95\%$.

---

## Installation

```bash
pip install torch numpy
```

---

## Running Tests

```bash
# Classifier accuracy
python run.py unittest classifier

# IoU calculation
python run.py unittest iou

# AUC calculation
python run.py unittest auc

# Raw FCN detector
python run.py unittest detector

# Non-Maximum Suppression
python run.py unittest nms

# Full detector with NMS (target AUC >= 95%)
python run.py unittest detector_nms
```
