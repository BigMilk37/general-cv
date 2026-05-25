# 📸 Image Demosaicing — Bayer Filter Interpolation

> Reconstruct full-color RGB images from single-channel Bayer RAW sensor data using classical and advanced CFA interpolation algorithms.

---

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Bayer Pattern](#bayer-pattern)
- [Usage & Tests](#usage--tests)
- [Implementation Details](#implementation-details)

---

## Overview

Digital camera sensors use a **Color Filter Array (CFA)** where each pixel captures only one color channel — Red, Green, or Blue. Demosaicing reconstructs the missing two channels at every pixel to produce a full RGB image.

This project implements and benchmarks two approaches:

| Algorithm | Window | Description |
|-----------|--------|-------------|
| **Bilinear Interpolation** | 3×3 | Classic baseline — averages neighboring pixels of the same channel |
| **Malvar-He-Cutler (MHC)** | 5×5 | Gradient-corrected linear filtering using cross-channel correlation |

---

## Algorithms

### Bilinear Interpolation

Reconstructs each missing channel by averaging all available neighbors of that color within a **3×3 window** around each pixel.

### Malvar-He-Cutler (MHC)

Improves on bilinear estimates by adding a **gradient correction term** derived from the known channel. For example, estimating Green at a Red pixel:

$$\hat{g}(i,j) = \hat{g}_{B}(i,j) + \alpha \cdot \Delta_{R}(i,j)$$

Where:

- $\hat{g}_{B}(i,j)$ — standard bilinear estimate of Green
- $\Delta_{R}(i,j)$ — localized spatial gradient of the Red channel
- $\alpha, \beta, \gamma$ — Wiener-optimized gain coefficients: $\alpha = \tfrac{1}{2}$, $\beta = \tfrac{5}{8}$, $\gamma = \tfrac{3}{4}$

> **Note:** The paper's filters are unnormalized — divide all coefficients by their sum to preserve global brightness.

---

## Project Structure

```
.
├── bayer.py          # Core implementation (your code goes here)
├── common.py         # Helper functions and test utilities
├── calibrate.py      # Calibrates time limits to local CPU speed
├── run.py            # Main test runner
└── tests/            # Automated test suites
```

---

## Installation

**Prerequisites:** Python 3.x

```bash
pip install numpy pytest
```

**Calibrate time limits** to your machine before running benchmarks:

```bash
python calibrate.py
```

This writes a `calibration.json` scale factor so performance tests are fair across different hardware.

---

## Bayer Pattern

This project targets the **GRBG** mosaic pattern:

```
G  R  G  R  ...
B  G  B  G  ...
G  R  G  R  ...
...
```

- **Row 0:** G R G R …
- **Row 1:** B G B G …

---

## Usage & Tests

Run individual test suites with:

```bash
./run.py unittest <test_name>
```

| Command | Step | Description |
|---------|------|-------------|
| `./run.py unittest masks` | 1 | `get_bayer_masks(n_rows, n_cols)` — binary R/G/B location matrices |
| `./run.py unittest colored_raw_img` | 2 | `get_colored_img` / `get_raw_img` — RAW ↔ 3D array conversion |
| `./run.py unittest bilinear` | 3 | Bilinear interpolation correctness on a 3×3 window |
| `./run.py unittest bilinear_img` | 3 | Visual validation on full-scale photos |
| `./run.py unittest bilinear_img_fast` | 3 | ⚡ Speed benchmark — must be fully vectorized |
| `./run.py unittest improved` | 4 | MHC algorithm correctness |
| `./run.py unittest improved_img` | 4 | Visual evaluation on high-resolution photos |
| `./run.py unittest improved_img_fast` | 4 | ⚡ Speed benchmark for MHC |
| `./run.py unittest psnr` | 5 | `compute_psnr(img_pred, img_gt)` — Peak Signal-to-Noise Ratio |

---

## Implementation Details

### Vectorization Strategy

Both algorithms avoid all pixel-level Python loops. Instead, neighbor accumulation is done via **NumPy slice-shift stencils**:

```python
# Accumulating vertical neighbors without loops
ans[:a - 1, :, 0] += r[1:a, :]
ans[1:,    :, 0] += r[0:a - 1, :]
```

### Boundary Handling

Rather than padding or branching, the code builds **dynamic divisor maps** that track how many valid neighbors each pixel actually has:

```python
# Count valid red neighbors per pixel dynamically
rcof[:a - 1, :] += ms[1:a, :, 0]
```

Dividing accumulated sums by these maps normalizes edge pixels automatically.

### Checkerboard Striding (MHC)

The alternating pixel structure of the 5×5 Malvar-He-Cutler kernel is handled with step slicing — no conditionals needed:

```python
# Isolate alternating grid positions
bcof[2::2, ::2] -= ms[:a - 2:2, ::2, 1]
```

### Processing Requirements

- **DataType Widening** — cast `uint8` input to `int32` or `float64` before convolutions to prevent overflow
- **Value Clipping** — clamp all output to `[0, 255]` with `np.clip(..., 0, 255)`
- **Filter Normalization** — divide MHC filter coefficients by their sum before applying
