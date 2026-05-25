# 🎞️ Prokudin-Gorsky Image Alignment

> Automatically align and reconstruct century-old color photographs from Sergey Prokudin-Gorsky's digitized glass-plate negatives using image pyramids and FFT phase correlation.

---

## Table of Contents

- [Background](#background)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage & Tests](#usage--tests)
- [Performance](#performance)

---

## Background

At the turn of the 20th century, Prokudin-Gorsky traveled across the Russian Empire photographing daily life through **Blue, Green, and Red filters** — capturing three consecutive exposures on a single vertical glass plate.

Because of wind, vibration, and time between shots, the three exposures are never perfectly aligned. This project automates finding the pixel offsets between channels to merge them into a single crisp RGB image.

```
┌───────────┐
│  Blue (B) │  ← top third
├───────────┤
│ Green (G) │  ← middle third
├───────────┤
│   Red (R) │  ← bottom third
└───────────┘
   raw plate
```

---

## Algorithms

### 1 · Channel Extraction

The plate is split into three equal vertical slices. With `crop=True`, the outer 10% of borders are removed to exclude glass edges and frame artifacts that would bias alignment.

### 2 · Image Pyramid (Spatial Domain)

Direct grid search on large images is too slow. The pyramid approach recursively downsamples by 2× until the image fits under 500px, then searches coarse-to-fine:

1. Find the best shift $(\Delta x, \Delta y)$ at the smallest scale using **MSE**:

$$\text{MSE}(I_1, I_2) = \frac{1}{W \cdot H} \sum_{x,y} \bigl(I_1(x,y) - I_2(x,y)\bigr)^2$$

2. Double the estimate and refine within a **±1 px neighborhood** at the next scale.
3. Repeat up to full resolution.

### 3 · Fourier Alignment (Frequency Domain)

Uses the **Fourier Correlation Theorem** to compute cross-correlation in $O(N \log N)$:

$$C(u,v) = \mathcal{F}^{-1}\!\left(\,\overline{\mathcal{F}\{I_1\}} \cdot \mathcal{F}\{I_2\}\right)$$

The peak of $|C(u,v)|$ gives the exact cyclic offset $(\Delta x, \Delta y)$ directly — no iterative search needed.

---

## Project Structure

```
.
├── align.py       # Core implementation — extraction, pyramid align, Fourier align
├── pipeline.py    # Orchestrates alignment, evaluation, and visualization
├── common.py      # Image I/O and test utilities
├── run.py         # Automated test runner
└── tests/         # Synthetic and real image test assets
```

---

## Installation

**Prerequisites:** Python 3.8+

```bash
pip install numpy pytest
```

---

## Usage & Tests

**Run the full alignment pipeline** on a sample plate (outputs `pyramid_aligned.png`, `fourier_aligned.png`, and debug visualizations):

```bash
python3 align.py
```

**Run individual test suites:**

```bash
# Channel extraction
python3 run.py unittest extract_channel_plates

# Spatial pyramid alignment
python3 run.py unittest find_relative_shift_pyramid

# Fourier frequency-domain alignment
python3 run.py unittest find_relative_shift_fourier

# End-to-end alignment on real plates
python3 run.py unittest align_image_pyramid_img_small
python3 run.py unittest align_image_pyramid_img_large
```

---

## Performance

| Method | Complexity | Accuracy on Large Scans | Best For |
|--------|-----------|------------------------|----------|
| Image Pyramid | Coarse-to-fine search | High | Standard plates with complex borders |
| Fourier (FFT) | $O(N \log N)$ | Very high | Large offsets, high-resolution scans |

> **Implementation note:** MSE was chosen over normalized cross-correlation for the pyramid search — benchmarking showed NCC was ~40% slower (13.9s vs 8.4s) with no meaningful accuracy gain on this dataset.

---

*Historical photographic plates courtesy of the [Library of Congress](https://www.loc.gov/collections/prokudin-gorsky/).*
