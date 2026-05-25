# NumPy Deep Learning Framework from Scratch

A clean, modular, zero-dependency (except NumPy) deep learning framework implemented from scratch. This project demonstrates the underlying mathematics of modern neural networks by implementing analytical forward and backward (backpropagation) passes for both Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs).

Developed as part of the Graphics & Media Lab (Vision Group) curriculum.

---

## Features

- **Custom Autograd Engine** — Analytical calculation of gradients using tensor derivatives without relying on modern deep learning frameworks (PyTorch/TensorFlow) for backpropagation.
- **Vectorized Layers** — High-performance vectorized operations leveraging NumPy's broadcasting, sliding window views, and Einstein summation (`np.einsum`).
- **Fully Connected & Convolutional Layers** — Implements modern operations including 2D convolutions, spatial pooling, and Batch Normalization.
- **Numerical Stability** — Built-in protection against overflow/underflow (e.g., shifted Softmax, epsilon-bounded logs).

---

## Architecture & Layers

### Optimizers

- **SGD** — Basic Stochastic Gradient Descent.
- **SGD with Momentum** — Accelerates gradient vectors in the right directions, reducing oscillation using a velocity momentum factor:

$$I \leftarrow \beta I + \alpha \frac{\partial L}{\partial P}$$

$$P \leftarrow P - I$$

### Standard Layers & Activation Functions

- **ReLU** — Rectified Linear Unit element-wise activation.
- **Softmax** — Vectorized, numerically stable multi-class probability generator:

$$y_i = \frac{e^{x_i - \max(X)}}{\sum e^{x_j - \max(X)}}$$

- **Dense (Fully Connected)** — Linear projection of activation inputs.

### Convolutional Architecture (CNN)

- **`convolve_numpy`** — A custom 2D multi-channel cross-correlation/convolution implementation utilizing sliding window views for spatial processing.
- **`Conv2D`** — Spatial convolutional layer with trainable kernels and biases, preserving dimensions through `same` padding.
- **`Pooling2D`** — Spatial downsampling layer supporting both Max Pooling (tracking indices for gradient routing) and Average Pooling.
- **`BatchNorm`** — Normalizes activations across batch, height, and width to stabilize training. Tracks rolling training metrics (`running_mean`, `running_var`) for inference mode.
- **`Flatten`** — Utility layer transitioning multi-dimensional feature maps into 1D vectors for linear classification heads.
- **`Dropout`** — Regularization method that randomly deactivates connections with probability $p$ during training, scaling inputs by $1-p$ during inference.

### Loss Function

- **Categorical Cross-Entropy** — Compares estimated probabilities against one-hot targets:

$$\text{Loss} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{d} y_{i,n}^{gt} \ln(y_{i,n}^{pred})$$

---

## Mathematical Derivations for Backpropagation

To update parameters and flow gradients backward through the computational graph, the analytical gradients of each layer are implemented as follows.

**ReLU Gradient**

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot [X \ge 0]$$

**Softmax Gradient**

$$\frac{\partial L}{\partial X} = Y \odot \frac{\partial L}{\partial Y} - Y \odot \left( \sum_{i} Y_i \frac{\partial L}{\partial Y_i} \right)$$

**Dense Layer Gradients**

$$\frac{\partial L}{\partial B} = \sum_{\text{batch}} \frac{\partial L}{\partial Y}$$

$$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Y}\right)^T \cdot X$$

$$\frac{\partial L}{\partial X} = W^T \cdot \frac{\partial L}{\partial Y}$$

**Convolution 2D Gradients**

Let $\hat{M}$ denote the tensor $M$ flipped along its spatial dimensions (180-degree rotation). The gradients are computed as:

$$\frac{\partial L}{\partial B} = \sum_{h, w} \frac{\partial L}{\partial Y}$$

$$\frac{\partial L}{\partial K} = \text{convolve}_{p}^*(\hat{X}^T,\ \frac{\partial L}{\partial Y}^T)^T$$

$$\frac{\partial L}{\partial X} = \text{convolve}_{p'}(\frac{\partial L}{\partial Y},\ \hat{K}^T)$$

---

## Model Performance & Benchmarks

### MNIST Handwritten Digit Classification

**Architecture**

```
Input(784) -> Dense(16) -> ReLU -> Dense(16) -> ReLU -> Dense(10) -> Softmax
```

| Metric | Target |
|---|---|
| Test Accuracy | > 90% |
| Training Time (CPU) | < 10 minutes |

### CIFAR-10 Image Classification

**Architecture**

```
Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
Conv2D(64, 3x3) -> BatchNorm -> Dropout(0.1) -> ReLU -> MaxPool(2x2)
Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
Flatten -> Dropout(0.4) -> Dense(64) -> Dense(10) -> Softmax
```

| Metric | Target |
|---|---|
| Test Accuracy | > 70% |
| Training Time (CPU) | < 60 minutes |

---

## Project Structure

```
├── common.py       # Internal utility classes & helper functions
├── interface.py    # Abstract Base Classes (Layer, Loss, Optimizer, Model)
├── run.py          # Test execution engine and training scripts
├── solution.py     # Core framework logic
└── tests/          # Local test suites for verification
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- PyTorch *(optional — recommended for faster CPU-less execution of `convolve` during model fitting)*

### Installation

```bash
git clone https://github.com/yourusername/numpy-dl-framework.git
cd numpy-dl-framework
pip install numpy
```

### Running Unit Tests

Verify individual implementations or run the full test suite via `run.py`:

```bash
# Optimizers
python run.py unittest sgd
python run.py unittest momentum

# Activations
python run.py unittest relu
python run.py unittest softmax

# Dense layers & loss
python run.py unittest dense
python run.py unittest crossentropy

# Convolutional blocks
python run.py unittest convolve
python run.py unittest conv2d
python run.py unittest pooling2d
python run.py unittest batchnorm
python run.py unittest flatten
python run.py unittest dropout
```

### Training the Models

```bash
# Train MLP on MNIST
python run.py train mnist

# Train CNN on CIFAR-10
python run.py train cifar10
```
