# 🔭 Image Deconvolution — Inverse & Wiener Filtering

> Restore blurred and noisy images in the frequency domain using inverse filtering with thresholding and optimal Wiener filtering.

---

## Table of Contents

- [Theory](#theory)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage & Tests](#usage--tests)
- [Visualization](#visualization)
- [Function Reference](#function-reference)

---

## Theory

Image degradation is modeled as convolution with a blur kernel plus additive noise:

$$g(x,y) = f(x,y) * h(x,y) + \eta(x,y)$$

| Symbol | Meaning |
|--------|---------|
| $f(x,y)$ | Original pristine image |
| $h(x,y)$ | Blur operator (Point Spread Function) |
| $\eta(x,y)$ | Additive noise |
| $g(x,y)$ | Observed degraded image |

By the **Convolution Theorem**, this becomes element-wise multiplication in the frequency domain:

$$G(u,v) = F(u,v) \cdot H(u,v) + N(u,v)$$

Deconvolution means recovering $F$ from the known $G$ and $H$.

---

## Algorithms

### 1 · Gaussian Blur Kernel

The distortion function is an isotropic 2D Gaussian:

$$h(r) = \frac{1}{2\pi\sigma^2}\, e^{-\frac{r^2}{2\sigma^2}}, \qquad r = \sqrt{(x-x_0)^2 + (y-y_0)^2}$$

The kernel is constructed symmetrically for both even and odd `size`, then normalized so coefficients sum to 1.

### 2 · Inverse Filtering

Estimates the original image by dividing out the blur in the frequency domain:

$$\tilde{F}(u,v) = G(u,v) \cdot H_{\text{inv}}(u,v)$$

$$H_{\text{inv}}(u,v) = \begin{cases} 0 & |H(u,v)| \le \varepsilon \\ \dfrac{1}{H(u,v)} & \text{otherwise} \end{cases}$$

The threshold $\varepsilon$ prevents near-zero values of $H$ from amplifying noise into unbounded artifacts. Works well on clean images; degrades heavily when noise is present.

### 3 · Wiener Filtering

Minimizes the mean squared error $E\bigl[(f - \tilde{f})^2\bigr]$ by treating signal and noise as random processes:

$$\tilde{F}(u,v) = \frac{\overline{H(u,v)}}{|H(u,v)|^2 + K}\, G(u,v)$$

| Term | Meaning |
|------|---------|
| $\overline{H(u,v)}$ | Complex conjugate of the blur kernel's Fourier transform |
| $\lvert H \rvert^2$ | Power spectrum of the blur |
| $K$ | Noise-to-signal ratio approximation |

A larger $K$ suppresses noise more aggressively at the cost of sharpness. Default: `K = 0.00009`.

### 4 · Quality Metric (PSNR)

$$\text{PSNR} = 20 \cdot \log_{10}\!\left(\frac{255}{\sqrt{\text{MSE}}}\right)$$

Higher PSNR = closer reconstruction to the original.

---

## Project Structure

```
.
├── deconvolution.py    # Core implementation
├── visualization.py    # Visualization scripts for inverse and Wiener filtering
├── run.py              # Test runner
└── tests/              # Unit test assets
```

---

## Installation

**Prerequisites:** Python 3.8+

```bash
pip install numpy scipy matplotlib
```

---

## Usage & Tests

```bash
python run.py unittest gaussian           # Gaussian kernel construction
python run.py unittest fourier_transform  # Kernel padding and FFT alignment
python run.py unittest inverse_kernel     # Threshold-based inversion
python run.py unittest inverse_filtering  # Full inverse filtering pipeline
python run.py unittest wiener_filtering   # Wiener filter restoration
python run.py unittest psnr               # PSNR metric
```

---

## Visualization

**Inverse filtering** — clean vs. noisy images across thresholds:

```bash
python visualization.py inverse
```

| Output | Description |
|--------|-------------|
| `inverse_filtering_blurred.jpg` | Near-perfect reconstruction on noise-free input |
| `inverse_filtering_noisy.jpg` | Noise amplification at various threshold values |

**Wiener filtering** — sweep over `K` to find the optimal noise-suppression constant for a 15×15 Gaussian kernel (σ = 5):

```bash
python visualization.py wiener
```

| Output | Description |
|--------|-------------|
| `wiener_filtering_noisy.jpg` | Restored image with significantly reduced noise vs. inverse filtering |

---

## Function Reference

| Function | Description |
|----------|-------------|
| `gaussian_kernel(size, sigma)` | Normalized symmetric 2D Gaussian filter |
| `fourier_transform(h, shape)` | Pads kernel to image size, returns centered 2D FFT |
| `inverse_kernel(H, threshold)` | Stable element-wise inversion with zero-thresholding |
| `inverse_filtering(blurred_img, h, threshold)` | Frequency-domain inverse filter restoration |
| `wiener_filtering(blurred_img, h, K)` | Wiener filter restoration (default `K=0.00009`) |
| `compute_psnr(img1, img2)` | Peak Signal-to-Noise Ratio between two images |
