# Generative Adversarial Networks in PyTorch

A PyTorch + Lightning implementation covering three progressive GAN tasks: low-resolution image generation, super-resolution upscaling, and quantitative evaluation of generative models using PRD and FID metrics. All experiments are run on a sneaker image dataset.

---

## Overview

The notebook is structured around three parts:

1. **Image Generation** — Train a fully connected GAN and a Deep Convolutional GAN (DCGAN) to generate $28 \times 28$ sneaker images from Gaussian noise.
2. **Super Resolution** — Train an SRGAN-style model to upscale $28 \times 28$ images to $112 \times 112$.
3. **GAN Metrics** — Evaluate generation quality using Precision-Recall Density (PRD) and Fréchet Inception Distance (FID).

---

## Part I: Image Generation

### Fully Connected GAN

A baseline GAN using only linear layers. Both the generator and discriminator operate on flattened image vectors.

**Generator architecture**

```
Linear(noise_dim, 1024) -> ReLU
Linear(1024, 1024)      -> ReLU
Linear(1024, 28*28*3)   -> reshape to (28, 28, 3)
```

**Discriminator architecture**

```
Flatten
Linear(28*28*3, 256) -> LeakyReLU(0.01)
Linear(256, 256)     -> LeakyReLU(0.01)
Linear(256, 1)
```

**Loss functions**

$$\ell_G = -\mathbb{E}_{z \sim p(z)} \left[ \log D(G(z)) \right]$$

$$\ell_D = -\mathbb{E}_{x \sim p_{\text{data}}} \left[ \log D(x) \right] - \mathbb{E}_{z \sim p(z)} \left[ \log(1 - D(G(z))) \right]$$

Losses are averaged over the minibatch. Both generator and discriminator are optimized with Adam ($lr=0.0002$, $\beta_1=0.5$, $\beta_2=0.999$), using manual optimization via Lightning.

---

### Deep Convolutional GAN (DCGAN)

Replaces linear layers with convolutional blocks to enable spatial reasoning.

**Generator architecture**

```
Linear(noise_dim, 128*7*7) -> Unflatten to (128, 7, 7)
ReLU
UpsamplingBilinear2d(x2) -> Conv2d(128, 128, 3) -> BatchNorm2d -> ReLU
UpsamplingBilinear2d(x2) -> Conv2d(128, 64, 3)  -> BatchNorm2d -> ReLU
Conv2d(64, 3, 3) -> Tanh
```

**Discriminator architecture**

```
Conv2d(3,  32, 3, stride=2) -> LeakyReLU(0.2) -> Dropout(0.25)
Conv2d(32, 64, 3, stride=2) -> ZeroPad2d      -> BatchNorm2d   -> LeakyReLU(0.2) -> Dropout(0.25)
Conv2d(64, 128, 3, stride=2)-> BatchNorm2d    -> LeakyReLU(0.2)-> Dropout(0.25)
Flatten -> Linear -> Sigmoid
```

**Least Squares GAN (LSGAN) loss**

An alternative, more stable loss replacing binary cross-entropy with mean squared error on discriminator outputs.

---

## Part II: Super Resolution

An SRGAN-style model trained to upscale $28 \times 28$ images to $112 \times 112$. MSE between real and generated high-resolution images is used as the pixel-level loss in place of VGG feature loss.

### Building Blocks

**Residual Block**

```
Conv2d(f, f, 3) -> ReLU -> BatchNorm2d
Conv2d(f, f, 3)          -> BatchNorm2d
+ skip connection
```

**Upsampling Block**

```
Upsample(scale_factor=2) -> Conv2d -> ReLU
```

### Generator (`SRGenerator`)

```
Conv2d(3, 64, 9) -> ReLU
16x ResidualBlock(64)
Conv2d(64, 64, 3) -> BatchNorm2d
+ skip connection from init_conv
2x UpsamplingBlock
Conv2d(64, 3, 9) -> Tanh
```

### Discriminator (`SRDiscriminator`)

```
DBlock(3,   64,  stride=1)
DBlock(64,  64,  stride=1)
DBlock(64,  128, stride=1)
DBlock(128, 128, stride=2)
DBlock(128, 256, stride=1)
DBlock(256, 256, stride=2)
DBlock(256, 512, stride=1)
DBlock(512, 512, stride=2)
Flatten -> Linear(512*196, 1024) -> Linear(1024, 1)
```

Each `DBlock` is `Conv2d -> LeakyReLU -> (optional) BatchNorm2d`.

---

## Part III: GAN Metrics

### Precision-Recall Density (PRD Score)

Measures both fidelity (precision) and diversity (recall) of generated samples. Features are clustered with k-means; precision and recall curves are swept over a set of thresholds $\theta$:

$$\alpha(\theta) = \sum_i \min(\tan(\theta) \cdot p_{\text{real},i},\ p_{\text{fake},i})$$
$$\beta(\theta) = \sum_i \min(p_{\text{real},i},\ \tan(\theta) \cdot p_{\text{fake},i})$$

The PRD score is the AUC of the resulting ($\beta$, $\alpha$) curve (recall vs. precision).

### Fréchet Inception Distance (FID)

Compares distributions of real and generated images in the feature space of a pretrained InceptionV3 model (pool3 layer, 2048-dim activations). Lower is better.

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

where $X_r \sim \mathcal{N}(\mu_r, \Sigma_r)$ and $X_g \sim \mathcal{N}(\mu_g, \Sigma_g)$ are the Inception activations for real and generated samples.

---

## Installation

```bash
pip install lightning torch torchvision scikit-learn scipy matplotlib opencv-python tqdm
```

### Dataset

```bash
curl -sO 'https://code.mipt.ru/courses-public/cv/storage/-/raw/tasks/sneaker-generation/data.zip'
unzip -qo data.zip
```

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Batch size | 128 |
| Noise dimension | 100 |
| Low-res image size | 28 × 28 |
| High-res image size | 112 × 112 |
| Learning rate | 0.0002 |
| Adam $\beta_1$ / $\beta_2$ | 0.5 / 0.999 |
| Training epochs (GAN) | 100 |
