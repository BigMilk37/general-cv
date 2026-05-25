# Multiple Object Tracking (MOT)

A PyTorch-based multiple object tracking pipeline combining a Single Shot Detector (SSD) for frame-level detection, an IoU-based tracklet association engine, and a Normalized Cross-Correlation (NCC) template matcher for efficient inter-frame propagation. Tracking quality is evaluated with standard MOTP and MOTA metrics.

Developed as part of the Graphics & Media Lab (Vision Group) curriculum.

---

## Pipeline

```
[Video Input]
      |
      v
[Every Nth frame?]
  |             |
 Yes            No
  |             |
  v             v
[SSD Detector]  [Cross-Correlation Tracker]
  |             |   (NCC template match +
  v             |    Gaussian spatial penalty)
[IoU Tracklet   |
 Association]   |
  |             |
  └──────┬──────┘
         v
  [Synchronized Trajectories]
         |
         v
  [MOTP / MOTA Evaluation]
```

On detection frames, bounding boxes from the SSD are matched to existing tracklets via IoU. On intermediate frames, the previous bounding box is tracked using normalized cross-correlation with a Gaussian spatial prior to suppress drift.

---

## Components

### Detection — `detectioncv.py`

Wraps a pretrained SSD network with a preprocessing and postprocessing pipeline.

- **Preprocessing** — Resizes frames to $300 \times 300$, converts RGB to BGR, and subtracts ImageNet channel means:

$$\mu_{\text{ImageNet}} = [103.939,\ 116.779,\ 123.68]$$

- **Postprocessing** — Filters detections by confidence threshold $\tau$, optionally restricts to a target label set (PASCAL VOC classes), and maps normalized coordinates back to pixel space.

### Tracklet Association — `tracker.py`

Maintains object identity across frames without graph optimization solvers.

- **Initialization** — Assigns unique tracklet IDs to all detections in the first frame.
- **Historical Lookup** — Scans the last $\delta$ frames (configurable via `lookup_tail_size`) in reverse chronological order to prevent ID overwriting across temporal gaps.
- **IoU Binding** — Greedily matches current detections to previous tracklets by maximum IoU overlap. Unmatched detections are registered as new tracklets.

### Cross-Correlation Tracker — `cross_correlation.py`

Replaces SSD inference on intermediate frames with efficient template matching.

- Converts frames to grayscale and extracts the previous bounding box as a template.
- Searches within a padded region ($P_{\text{pad}} = 20\text{px}$) around the last known position.
- Weights the NCC response map by a Gaussian centered on the previous location to penalize large spatial jumps:

$$\text{bbox}^* = \arg\max \left[ \text{NCC}(\text{bbox},\ \text{prev\_bbox}) \times \mathcal{N}(\mu_{\text{center}},\ \sigma) \right]$$

### Evaluation Metrics — `metricscv.py`

Implements MOTP and MOTA from scratch, accumulating errors globally across all frames before computing ratios to avoid per-frame skew.

**Intersection over Union**

$$\text{IoU}(B_1, B_2) = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$

**Multiple Object Tracking Precision (MOTP)** — measures localization quality of matched pairs:

$$\text{MOTP} = \frac{\sum_{t,i} d_t^i}{\sum_t c_t}$$

where $d_t^i$ is the IoU distance for matched pair $i$ at frame $t$, and $c_t$ is the total number of matches at frame $t$.

**Multiple Object Tracking Accuracy (MOTA)** — measures tracking correctness across three error types:

$$\text{MOTA} = 1 - \frac{\sum_t (m_t + fp_t + mme_t)}{\sum_t g_t}$$

where $m_t$ = misses, $fp_t$ = false positives, $mme_t$ = identity mismatches, and $g_t$ = total ground truth objects at frame $t$.

---

## Repository Structure

```
├── detectioncv.py        # SSD detection wrapper and preprocessing
├── tracker.py            # IoU-based tracklet association
├── cross_correlation.py  # NCC template matching tracker
├── metricscv.py          # MOTP and MOTA metric implementations
├── config.py             # Model config, VOC class map, bbox utilities
├── utils.py              # Color mapping and visualization helpers
└── data/
    └── test.mp4          # Sample input video
```

---

## Installation

```bash
pip install numpy torch pillow scikit-image opencv-python moviepy
```

---

## Running Tests

```bash
# SSD detection pipeline
python run.py unittest detection

# IoU computation
python run.py unittest iou

# Tracklet association
python run.py unittest tracker

# MOTP metric
python run.py unittest motp

# MOTP + MOTA combined evaluation
python run.py unittest tracking
```

---

## Running the Tracker

```python
from moviepy.editor import VideoFileClip
from cross_correlation import CorrelationTracker

tracker = CorrelationTracker(detection_rate=5, labels=["car", "person"])
clip = VideoFileClip("data/test.mp4")
clip.fl_image(tracker.update_frame).preview()
```

`detection_rate=5` runs the SSD every 5 frames; NCC handles the frames in between.
