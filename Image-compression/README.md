# 🗜️ Image Compression — PCA & JPEG

> A from-scratch implementation and comparative study of PCA-based dimensionality reduction and the JPEG baseline pipeline, with PSNR and compression ratio analysis.

---

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage & Tests](#usage--tests)
- [Visualizations](#visualizations)
- [Function Reference](#function-reference)

---

## Overview

Two compression approaches are implemented and benchmarked:

| Method | Strategy | Control Parameter |
|--------|----------|-------------------|
| **PCA** | Projects color channels onto top eigenvectors | Number of components `p` |
| **JPEG** | DCT + quantization + RLE in YCbCr space | Quality factor `Q ∈ [1, 100]` |

Both pipelines output a reconstructed image alongside **PSNR** and **compression ratio** metrics, enabling direct quality-vs-size tradeoff comparison.

---

## Algorithms

### PCA Compression

Each color channel (R, G, B) is treated as a 2D matrix. Compression proceeds by:

1. **Centering** — subtract per-row means
2. **Covariance** — compute the sample covariance matrix
3. **Eigendecomposition** — extract eigenvectors via `scipy.linalg.eigh` (symmetric solver)
4. **Projection** — keep only the top `p` eigenvectors; discard the rest

Decompression multiplies the low-dimensional projections back into pixel space and adds row means.

### JPEG Pipeline

```
RGB → YCbCr → Downsample Cb/Cr → 8×8 DCT → Quantize → Zigzag → RLE
                                                                    ↕
RGB ← YCbCr ← Upsample Cb/Cr  ← 8×8 IDCT ← Dequantize ← Inverse zigzag ← Inverse RLE
```

**Why YCbCr?** Human vision is more sensitive to luma (Y) than chrominance (Cb, Cr), so chroma channels can be halved in resolution with minimal perceptual loss.

#### DCT (Forward)

$$G_{u,v} = \frac{1}{4}\,\alpha(u)\,\alpha(v) \sum_{x=0}^{7}\sum_{y=0}^{7} g_{x,y} \cos\!\left[\frac{(2x+1)\,u\pi}{16}\right] \cos\!\left[\frac{(2y+1)\,v\pi}{16}\right]$$

$$\alpha(u) = \begin{cases} \tfrac{1}{\sqrt{2}} & u = 0 \\ 1 & \text{otherwise} \end{cases}$$

#### Quality Factor → Quantization Matrix

$$S = \begin{cases} 5000/Q & 1 \le Q < 50 \\ 200 - 2Q & 50 \le Q \le 99 \\ 1 & Q = 100 \end{cases} \qquad Q_{i,j} = \left\lfloor \frac{50 + S \times D_{i,j}}{100} \right\rfloor$$

Any zero entry in $Q_{i,j}$ is replaced with 1 to avoid division by zero on decompression.

#### Zigzag + RLE

The quantized 8×8 block is read in a diagonal sweep to cluster the high-frequency near-zero coefficients at the end of the sequence, maximizing zero-run lengths before RLE encoding.

```
 0  1  5  6 14 15 27 28
 2  4  7 13 16 26 29 42
 3  8 12 17 25 30 41 43
 9 11 18 24 31 40 44 53
10 19 23 32 39 45 52 54
20 22 33 38 46 51 55 60
21 34 37 47 50 56 59 61
35 36 48 49 57 58 62 63
```

---

## Project Structure

```
.
├── image_compression.py   # Full PCA + JPEG implementation
├── description.pdf        # Formal specification (Russian)
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.8+

```bash
pip install numpy scipy scikit-image matplotlib
```

---

## Usage & Tests

```bash
# PCA
python run.py unittest pca_compression
python run.py unittest pca_decompression

# Color space
python run.py unittest rgb2ycbcr2rgb

# JPEG components
python run.py unittest downsampling
python run.py unittest dct
python run.py unittest quantization
python run.py unittest own_quantization
python run.py unittest zigzag
python run.py unittest compression
```

---

## Visualizations

Uncomment the `__main__` block in `image_compression.py` to generate all diagnostic outputs:

```python
pca_visualize()           # PCA reconstruction at p = 1, 5, 10, 20, 50, 100, 150, 200, 256
get_gauss_1()             # Gaussian blur on Cb/Cr (chroma) — shows chroma insensitivity
get_gauss_2()             # Gaussian blur on Y (luma) — shows luma sensitivity
jpeg_visualize()          # JPEG reconstruction at Q = 1, 10, 20, 50, 80, 100
get_pca_metrics_graph()   # PSNR and compression ratio curves for PCA
get_jpeg_metrics_graph()  # PSNR and compression ratio curves for JPEG
```

| Output file | Contents |
|-------------|----------|
| `pca_visualization.png` | 3×3 grid of PCA reconstructions at increasing `p` |
| `gauss_1.png` | Chroma-blurred image (visually mild degradation) |
| `gauss_2.png` | Luma-blurred image (visually severe degradation) |
| `jpeg_visualization.png` | 2×3 grid of JPEG reconstructions at increasing `Q` |
| `pca_metrics_graph.png` | Quality factor vs PSNR + PSNR vs compression ratio |
| `jpeg_metrics_graph.png` | Same metrics for JPEG |

---

## Function Reference

### PCA

| Function | Description |
|----------|-------------|
| `pca_compression(matrix, p)` | Returns `(eigenvectors, projections, row_means)` for one channel |
| `pca_decompression(compressed)` | Reconstructs RGB image from per-channel compressed tuples |

### JPEG

| Function | Description |
|----------|-------------|
| `rgb2ycbcr(img)` / `ycbcr2rgb(img)` | Color space conversion |
| `downsampling(component)` / `upsampling(component)` | Gaussian-filtered 2× chroma down/upsampling |
| `dct(block)` / `inverse_dct(block)` | Forward/inverse 2D DCT on 8×8 blocks |
| `quantization(block, matrix)` / `inverse_quantization(block, matrix)` | Frequency coefficient scaling |
| `own_quantization_matrix(default, q)` | Generates a quality-scaled quantization matrix |
| `zigzag(block)` / `inverse_zigzag(input)` | Diagonal scan ↔ 64-element vector |
| `compression(zigzag_list)` / `inverse_compression(compressed_list)` | Zero-run RLE encode/decode |
| `jpeg_compression(img, matrices)` | Full JPEG encode |
| `jpeg_decompression(result, shape, matrices)` | Full JPEG decode |
