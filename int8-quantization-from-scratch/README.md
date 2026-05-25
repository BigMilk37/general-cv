# Manual INT8 Quantization of a CNN from Scratch

A from-scratch implementation of post-training static INT8 quantization applied to a SimpleNet CNN trained on MNIST. The quantization scheme mirrors the TensorFlow Lite integer-arithmetic-only inference pipeline — all calibration, scaling, and fixed-point arithmetic are implemented manually in NumPy without relying on framework quantization APIs.

Developed as part of the Graphics & Media Lab (Vision Group) curriculum.

---

## Objectives

- Implement the full quantization pipeline by hand: parameter calibration, forward quantization, dequantization, and fixed-point multiplier arithmetic.
- Reduce model weight storage by 4× (FP32 → INT8) while preparing the compute graph for integer-only inference (INT8 weights/activations, INT32 accumulators).
- Measure accuracy degradation relative to the original FP32 baseline.

---

## Theory

Quantization maps a continuous real-valued range $[R_{\min}, R_{\max}]$ to a bounded integer range $[Q_{\min}, Q_{\max}]$ using a scale $S$ and zero-point $Z$.

### Scale and Zero-Point

$$S = \frac{R_{\max} - R_{\min}}{Q_{\max} - Q_{\min}}$$

$$Z = \text{round}\left(\frac{R_{\max} Q_{\min} - R_{\min} Q_{\max}}{R_{\max} - R_{\min}}\right)$$

$S \in$ `np.float64`, $Z \in$ `np.int32`.

### Forward Quantization

$$q(r) = \text{clip}\left(\text{round}\left(\frac{r}{S} + Z\right),\ Q_{\min},\ Q_{\max}\right)$$

### Dequantization

$$\tilde{r} = S(q - Z)$$

### Weight Quantization (Symmetric)

Weights are quantized symmetrically around zero ($Z_w = 0$) using the clipped range $[-127, 127]$.

- **Per-tensor** — one shared $S_w$ for the entire weight tensor.
- **Per-channel** — individual $S_w^{(c)}$ per output filter.

### Fixed-Point Multiplier

To avoid floating-point operations during inference, the real-valued scaling factor $M = \frac{S_w S_x}{S_y}$ is decomposed as:

$$M \approx 2^{-n} M_0$$

where $n \in$ `np.int32` and $M_0 \in [2^{30}, 2^{31} - 1]$ is a 32-bit fixed-point value encoding $M_0 \in [0.5, 1)$.

---

## Repository Structure

```
├── solution.py                  # Quantization algorithm implementations
├── fp32_model.py                # Trains and saves the FP32 SimpleNet baseline
├── manual_quantization.py       # Assembles the quantized model from solution.py
├── comparison.py                # Accuracy comparison across all model variants
├── run.py                       # Unit test runner
├── tensorflow_quantization.py   # (Optional) TFLite automatic quantization
└── pytorch_quantization.py      # (Optional) PyTorch automatic quantization
```

---

## Usage

### 1. Train the FP32 Baseline

```bash
python fp32_model.py
```

Trains SimpleNet on MNIST and saves `fp32_model.pt`.

### 2. Run Unit Tests

```bash
chmod +x run.py

./run.py unittest compute_quantization_params   # Scale and zero-point
./run.py unittest quantize                      # Forward quantization
./run.py unittest dequantize                    # Dequantization
./run.py unittest minmax_observer               # Calibration observer
./run.py unittest quantize_weights_per_tensor   # Per-tensor weight quantization
./run.py unittest quantize_weights_per_channel  # Per-channel weight quantization
./run.py unittest quantize_bias                 # Bias quantization
./run.py unittest quantize_multiplier           # Fixed-point multiplier encoding
./run.py unittest multiply_by_quantized_multiplier  # Fixed-point multiply
```

### 3. Compare Model Accuracy

```bash
python comparison.py
```

Evaluates the FP32 model, manual INT8 model, and (optionally) framework-quantized variants on MNIST. By default runs on 1000 samples — set `NUM_INPUTS = 10000` inside `comparison.py` for full evaluation.

---

## Implementation Summary

| Component | Description |
|---|---|
| `QuantizationParameters` | Dataclass holding $S$, $Z$, $Q_{\min}$, $Q_{\max}$ |
| `compute_quantization_params` | Computes $S$ and $Z$ from a real-valued range |
| `quantize` / `dequantize` | Forward and inverse quantization |
| `MinMaxObserver` | Tracks running min/max for calibration |
| `quantize_weights_per_tensor` | Symmetric per-tensor weight quantization |
| `quantize_weights_per_channel` | Symmetric per-channel weight quantization |
| `quantize_bias` | Quantizes bias into INT32 using $S_w \cdot S_x$ |
| `quantize_multiplier` | Encodes $M$ as $(n, M_0)$ fixed-point pair |
| `multiply_by_quantized_multiplier` | Integer-only multiply using $(n, M_0)$ |
