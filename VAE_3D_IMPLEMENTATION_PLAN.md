# Implementation Plan: 3D Variational Autoencoder (VAE) for VTEA

**Version:** 1.0
**Date:** 2025-12-26
**Branch:** `claude/add-3d-vae-vtea-KHwVA`
**Prerequisites:** 3D Deep Learning Infrastructure (DEEP_LEARNING_IMPLEMENTATION_PLAN.md)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [VAE Architecture Overview](#vae-architecture)
3. [Relationship to 3D Classification Infrastructure](#infrastructure-dependency)
4. [Implementation Plan Overview](#implementation-overview)
5. [Phase 1: VAE Core Architecture](#phase-1)
6. [Phase 2: Loss Functions & Training](#phase-2)
7. [Phase 3: Latent Space Analysis](#phase-3)
8. [Phase 4: Reconstruction & Quality Metrics](#phase-4)
9. [Phase 5: Dimensionality Reduction Integration](#phase-5)
10. [Phase 6: Anomaly Detection](#phase-6)
11. [Phase 7: UI Components](#phase-7)
12. [Implementation Sequence](#implementation-sequence)
13. [Use Cases in VTEA](#use-cases)
14. [Testing Strategy](#testing)
15. [Potential Challenges & Solutions](#challenges)

---

<a name="executive-summary"></a>
## 1. Executive Summary

### What is a VAE?

A **Variational Autoencoder (VAE)** is a generative deep learning model that learns to:
1. **Encode** high-dimensional data (3D cell images) into a low-dimensional latent space
2. **Decode** latent representations back to reconstructed images
3. Learn a **probabilistic distribution** in latent space (enabling generation and interpolation)
4. Provide **uncertainty quantification** for representations

### Why VAE for VTEA?

**Advantages over traditional methods:**
- **Unsupervised learning**: No manual labels required (unlike classification)
- **Dimensionality reduction**: Maps 3D volumes to 2D-50D latent space (vs. PCA/t-SNE on features)
- **Quality control**: Reconstruction error identifies poorly segmented cells
- **Anomaly detection**: Identifies rare or novel cell types
- **Data generation**: Can synthesize new cell examples for augmentation
- **Continuous representations**: Smooth latent space enables interpolation between cell types
- **Visualization**: 2D/3D latent space projections for exploration

**Use Cases:**
1. **Exploratory analysis**: Discover cell populations without manual gating
2. **Quality filtering**: Remove artifacts based on reconstruction quality
3. **Feature learning**: Use latent space as input to downstream classification
4. **Rare cell detection**: Identify outliers in latent space
5. **Data augmentation**: Generate synthetic training examples
6. **Batch correction**: Align datasets in latent space

### Integration with VTEA Workflow

```
3D Segmentation → Cell Regions → VAE Encoder → Latent Space
                                              ↓
                                    2D/3D Visualization
                                    Clustering (K-means on latent)
                                    Anomaly scores
                                    Classification (latent features)
                                              ↓
                                    VAE Decoder → Reconstructions
                                              ↓
                                    Quality metrics
                                    Visual inspection
```

---

<a name="vae-architecture"></a>
## 2. VAE Architecture Overview

### 2.1 Mathematical Foundation

**Encoder (Recognition Network):**
```
q(z|x) ~ N(μ(x), σ²(x))

where:
- x: Input 3D volume (e.g., 64×64×64)
- z: Latent vector (e.g., 32-dimensional)
- μ(x): Mean vector from encoder
- σ²(x): Variance vector from encoder
```

**Reparameterization Trick:**
```
z = μ + σ ⊙ ε,  where ε ~ N(0, I)

Enables backpropagation through sampling
```

**Decoder (Generative Network):**
```
p(x|z) ~ N(x_reconstructed, I)

Maps latent vector z back to 3D volume
```

**Loss Function:**
```
Loss = Reconstruction Loss + β × KL Divergence

Reconstruction Loss: MSE or Binary Cross-Entropy
KL Divergence: KL(q(z|x) || p(z)), where p(z) = N(0, I)
β: Weighting factor (β-VAE for disentanglement)
```

### 2.2 3D VAE Architecture for VTEA

**Encoder Network:**
```
Input: [B, C, D, H, W]  (Batch, Channels, Depth, Height, Width)
      ↓
Conv3D Block 1: C → 32 features
  - Conv3d(3×3×3, stride=2, padding=1) + BatchNorm + LeakyReLU
  - Conv3d(3×3×3, padding=1) + BatchNorm + LeakyReLU
      ↓
Conv3D Block 2: 32 → 64 features
  - Conv3d(3×3×3, stride=2, padding=1) + BatchNorm + LeakyReLU
  - Conv3d(3×3×3, padding=1) + BatchNorm + LeakyReLU
      ↓
Conv3D Block 3: 64 → 128 features
  - Conv3d(3×3×3, stride=2, padding=1) + BatchNorm + LeakyReLU
  - Conv3d(3×3×3, padding=1) + BatchNorm + LeakyReLU
      ↓
Conv3D Block 4: 128 → 256 features
  - Conv3d(3×3×3, stride=2, padding=1) + BatchNorm + LeakyReLU
      ↓
Flatten
      ↓
FC → μ (latent_dim)
FC → log_σ² (latent_dim)
```

**Latent Space:**
```
Sample: z ~ N(μ, exp(log_σ²))
Dimensionality: 16, 32, 64, 128 (configurable)
```

**Decoder Network (Mirror of Encoder):**
```
Input: z (latent_dim)
      ↓
FC → Reshape to [256, D', H', W']
      ↓
TransposeConv3D Block 1: 256 → 128 features
  - TransposeConv3d(3×3×3, stride=2, padding=1, output_padding=1)
  - BatchNorm + LeakyReLU
      ↓
TransposeConv3D Block 2: 128 → 64 features
  - TransposeConv3d(3×3×3, stride=2, padding=1, output_padding=1)
  - BatchNorm + LeakyReLU
      ↓
TransposeConv3D Block 3: 64 → 32 features
  - TransposeConv3d(3×3×3, stride=2, padding=1, output_padding=1)
  - BatchNorm + LeakyReLU
      ↓
TransposeConv3D Block 4: 32 → C features
  - TransposeConv3d(3×3×3, stride=2, padding=1, output_padding=1)
  - Sigmoid (for normalized inputs [0,1])
      ↓
Output: [B, C, D, H, W]  (Reconstructed volume)
```

### 2.3 Architecture Variants

**VAE3D_Small** (Fast, low memory):
- Input: 32³ volumes
- Latent: 16 dimensions
- Channels: [16, 32, 64, 128]
- Parameters: ~1.2M

**VAE3D_Medium** (Balanced):
- Input: 64³ volumes
- Latent: 32 dimensions
- Channels: [32, 64, 128, 256]
- Parameters: ~8.5M

**VAE3D_Large** (High quality):
- Input: 128³ volumes
- Latent: 64 dimensions
- Channels: [32, 64, 128, 256, 512]
- Parameters: ~35M

**β-VAE** (Disentangled representations):
- Same architectures as above
- β ∈ [1, 10] for disentanglement control
- Better for interpretable features

**Conditional VAE (cVAE)**:
- Adds class labels to encoder/decoder
- Enables class-specific generation
- Useful for multi-class datasets

---

<a name="infrastructure-dependency"></a>
## 3. Relationship to 3D Classification Infrastructure

### 3.1 Shared Components (From DEEP_LEARNING_IMPLEMENTATION_PLAN.md)

**✅ Reusable Components:**
1. **Dependencies** (pom.xml)
   - PyTorch JavaCPP bindings
   - GSON for configuration
   - CUDA support (optional)

2. **Data Pipeline** (`vtea/deeplearning/data/`)
   - `TensorConverter.java`: ImageStack ↔ PyTorch Tensor
   - `CellRegionExtractor.java`: Extract 3D regions from MicroObjects
   - `DatasetDefinition.java`: Configuration for datasets
   - Normalization strategies: Z-score, Min-Max

3. **Training Infrastructure** (`vtea/deeplearning/training/`)
   - `DataLoader.java`: Batching, shuffling
   - `ModelCheckpoint.java`: Save/load models
   - Training utilities: device management, logging

4. **Base Classes** (`vtea/deeplearning/models/`)
   - `AbstractDeepLearningModel.java`: Common model interface
   - Device management (CPU/GPU)
   - Parameter initialization

5. **UI Framework** (`vtea/exploration/plottools/panels/`)
   - ManualClassification.java (for labeled data if needed)
   - Plotting infrastructure for visualizations

### 3.2 VAE-Specific New Components

**🆕 New Classes:**
1. **Models** (`vtea/deeplearning/models/`)
   - `VariationalAutoencoder3D.java`: Main VAE implementation
   - `VAEEncoder3D.java`: Encoder network
   - `VAEDecoder3D.java`: Decoder network
   - `BetaVAE3D.java`: β-VAE variant
   - `ConditionalVAE3D.java`: cVAE variant

2. **Loss Functions** (`vtea/deeplearning/loss/`)
   - `VAELoss.java`: Combined reconstruction + KL divergence
   - `ReconstructionLoss.java`: MSE, BCE options
   - `KLDivergenceLoss.java`: Analytical KL for Gaussians
   - `BetaVAELoss.java`: Weighted KL divergence

3. **Training** (`vtea/deeplearning/training/`)
   - `VAETrainer.java`: Custom training loop for VAE
   - `VAEMetrics.java`: Track reconstruction, KL, ELBO

4. **Inference** (`vtea/deeplearning/inference/`)
   - `VAELatentExtractor.java`: Extract latent representations
   - `VAEReconstructor.java`: Generate reconstructions
   - `VAEAnomalyDetector.java`: Compute anomaly scores
   - `VAEGenerator.java`: Sample from latent space

5. **Analysis** (`vtea/deeplearning/analysis/`)
   - `LatentSpaceVisualizer.java`: 2D/3D projections (t-SNE, UMAP on latent)
   - `ReconstructionQualityAnalyzer.java`: Quality metrics
   - `LatentSpaceInterpolator.java`: Interpolate between cells
   - `DisentanglementMetrics.java`: For β-VAE

6. **Integration** (`vtea/featureprocessing/`)
   - `VAEFeatureExtraction.java`: Plugin to extract latent features
   - `VAEAnomalyDetection.java`: Plugin for quality control
   - `VAEClustering.java`: Cluster in latent space

### 3.3 Implementation Dependencies

**Sequence:**
```
Phase 0 (Prerequisites - from 3D Classification Plan):
  - PyTorch dependencies added to pom.xml
  - TensorConverter.java implemented
  - CellRegionExtractor.java implemented
  - DataLoader.java implemented
  - ModelCheckpoint.java implemented
  - AbstractDeepLearningModel.java implemented
        ↓
Phase 1 (VAE Core):
  - VAEEncoder3D.java
  - VAEDecoder3D.java
  - VariationalAutoencoder3D.java
        ↓
Phase 2 (Training):
  - VAELoss.java
  - VAETrainer.java
  - VAEMetrics.java
        ↓
Phase 3 (Analysis):
  - VAELatentExtractor.java
  - LatentSpaceVisualizer.java
        ↓
Phase 4-7 (Advanced features)
```

**Timeline Assumption:**
- If 3D Classification infrastructure is already implemented: **8-10 weeks**
- If starting from scratch (both projects): **22-26 weeks**

---

<a name="implementation-overview"></a>
## 4. Implementation Plan Overview

### 4.1 Goals

1. ✅ Implement 3D VAE for unsupervised learning on cell volumes
2. ✅ Enable dimensionality reduction to low-dimensional latent space
3. ✅ Provide reconstruction-based quality control
4. ✅ Support anomaly detection for rare cells
5. ✅ Integrate latent representations into VTEA workflows
6. ✅ Enable latent space visualization and exploration
7. ✅ Support multiple VAE variants (standard, β-VAE, cVAE)
8. ✅ Maintain compatibility with existing VTEA features

### 4.2 Package Structure

```
vtea/
├── deeplearning/
│   ├── models/
│   │   ├── AbstractDeepLearningModel.java         [FROM 3D CLASSIFICATION]
│   │   ├── NephNet3D.java                         [FROM 3D CLASSIFICATION]
│   │   ├── Generic3DCNN.java                      [FROM 3D CLASSIFICATION]
│   │   ├── VariationalAutoencoder3D.java          [NEW - VAE]
│   │   ├── VAEEncoder3D.java                      [NEW - VAE]
│   │   ├── VAEDecoder3D.java                      [NEW - VAE]
│   │   ├── BetaVAE3D.java                         [NEW - VAE]
│   │   └── ConditionalVAE3D.java                  [NEW - VAE]
│   ├── data/
│   │   ├── TensorConverter.java                   [FROM 3D CLASSIFICATION]
│   │   ├── DatasetDefinition.java                 [FROM 3D CLASSIFICATION]
│   │   ├── CellRegionExtractor.java               [FROM 3D CLASSIFICATION]
│   │   └── VAEDataset.java                        [NEW - VAE]
│   ├── loss/
│   │   ├── VAELoss.java                           [NEW - VAE]
│   │   ├── ReconstructionLoss.java                [NEW - VAE]
│   │   ├── KLDivergenceLoss.java                  [NEW - VAE]
│   │   └── BetaVAELoss.java                       [NEW - VAE]
│   ├── training/
│   │   ├── Trainer.java                           [FROM 3D CLASSIFICATION]
│   │   ├── DataLoader.java                        [FROM 3D CLASSIFICATION]
│   │   ├── ModelCheckpoint.java                   [FROM 3D CLASSIFICATION]
│   │   ├── VAETrainer.java                        [NEW - VAE]
│   │   └── VAEMetrics.java                        [NEW - VAE]
│   ├── inference/
│   │   ├── DeepLearningClassification.java        [FROM 3D CLASSIFICATION]
│   │   ├── VAELatentExtractor.java                [NEW - VAE]
│   │   ├── VAEReconstructor.java                  [NEW - VAE]
│   │   ├── VAEAnomalyDetector.java                [NEW - VAE]
│   │   └── VAEGenerator.java                      [NEW - VAE]
│   └── analysis/
│       ├── LatentSpaceVisualizer.java             [NEW - VAE]
│       ├── ReconstructionQualityAnalyzer.java     [NEW - VAE]
│       ├── LatentSpaceInterpolator.java           [NEW - VAE]
│       └── DisentanglementMetrics.java            [NEW - VAE]
├── featureprocessing/
│   ├── AbstractFeatureProcessing.java             [EXISTING]
│   ├── VAEFeatureExtraction.java                  [NEW - VAE Plugin]
│   ├── VAEAnomalyDetection.java                   [NEW - VAE Plugin]
│   └── VAEClustering.java                         [NEW - VAE Plugin]
└── exploration/
    └── plottools/
        └── panels/
            ├── VAELatentSpacePanel.java           [NEW - VAE UI]
            ├── VAEReconstructionPanel.java        [NEW - VAE UI]
            └── VAETrainingPanel.java              [NEW - VAE UI]
```

### 4.3 Configuration Management

**VAEConfig.java:**
```java
public class VAEConfig implements Serializable {
    // Architecture
    private VAEArchitecture architecture;  // SMALL, MEDIUM, LARGE
    private int latentDim;                 // 16, 32, 64, 128
    private int[] encoderChannels;         // [32, 64, 128, 256]
    private int inputSize;                 // 32, 64, 128

    // Training
    private double learningRate;           // 1e-4
    private int batchSize;                 // 16, 32
    private int epochs;                    // 100
    private double beta;                   // 1.0 (for β-VAE)

    // Loss
    private ReconstructionType reconType;  // MSE, BCE
    private double klWeight;               // 1.0 or beta value
    private boolean warmupKL;              // Gradual KL weighting
    private int warmupEpochs;              // 10

    // Data
    private NormalizationType normalization;
    private int numChannels;
    private boolean useAugmentation;

    // Inference
    private int numSamples;                // For uncertainty estimation
    private double anomalyThreshold;       // Percentile for anomalies
}
```

---

<a name="phase-1"></a>
## 5. Phase 1: VAE Core Architecture

### 5.1 VAEEncoder3D.java

**Responsibilities:**
- Encode 3D volumes to latent distribution parameters (μ, log σ²)
- Convolutional feature extraction
- Fully connected layers to latent space

**Implementation:**
```java
package vtea.deeplearning.models;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class VAEEncoder3D extends Module {
    private final Sequential convBlocks;
    private final Linear fcMu;
    private final Linear fcLogVar;
    private final int latentDim;

    public VAEEncoder3D(int inputChannels, int latentDim, int[] channels) {
        this.latentDim = latentDim;

        // Build convolutional blocks
        this.convBlocks = buildConvBlocks(inputChannels, channels);

        // Calculate flattened size after convolutions
        int flattenedSize = calculateFlattenedSize(channels);

        // Latent space projection
        this.fcMu = new Linear(flattenedSize, latentDim);
        this.fcLogVar = new Linear(flattenedSize, latentDim);

        // Register modules
        register_module("conv_blocks", convBlocks);
        register_module("fc_mu", fcMu);
        register_module("fc_logvar", fcLogVar);

        // Initialize weights
        initializeWeights();
    }

    public EncoderOutput forward(Tensor x) {
        // Convolutional feature extraction
        Tensor features = convBlocks.forward(x);

        // Flatten
        Tensor flattened = features.view(new long[]{features.size(0), -1});

        // Latent parameters
        Tensor mu = fcMu.forward(flattened);
        Tensor logVar = fcLogVar.forward(flattened);

        return new EncoderOutput(mu, logVar);
    }

    private Sequential buildConvBlocks(int inChannels, int[] channels) {
        Sequential blocks = new Sequential();
        int currentChannels = inChannels;

        for (int i = 0; i < channels.length; i++) {
            blocks.add(createConvBlock(currentChannels, channels[i],
                                      i < channels.length - 1)); // stride=2 except last
            currentChannels = channels[i];
        }

        return blocks;
    }

    private Sequential createConvBlock(int in, int out, boolean downsample) {
        Sequential block = new Sequential();
        int stride = downsample ? 2 : 1;

        // Conv3d + BatchNorm + LeakyReLU
        block.add(Conv3d(in, out, new long[]{3, 3, 3},
                        stride, new long[]{1, 1, 1}));
        block.add(BatchNorm3d(out));
        block.add(LeakyReLU(0.2));

        // Second conv (no downsampling)
        block.add(Conv3d(out, out, new long[]{3, 3, 3},
                        1, new long[]{1, 1, 1}));
        block.add(BatchNorm3d(out));
        block.add(LeakyReLU(0.2));

        return block;
    }

    private void initializeWeights() {
        // Kaiming normal initialization for conv layers
        // Normal initialization for linear layers
    }

    public static class EncoderOutput {
        public final Tensor mu;
        public final Tensor logVar;

        public EncoderOutput(Tensor mu, Tensor logVar) {
            this.mu = mu;
            this.logVar = logVar;
        }
    }
}
```

### 5.2 VAEDecoder3D.java

**Responsibilities:**
- Decode latent vectors to 3D volumes
- Transpose convolutions for upsampling
- Mirror encoder architecture

**Implementation:**
```java
package vtea.deeplearning.models;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class VAEDecoder3D extends Module {
    private final Linear fcProject;
    private final Sequential deconvBlocks;
    private final int latentDim;
    private final int outputChannels;
    private final long[] reshapeSize;

    public VAEDecoder3D(int latentDim, int outputChannels,
                       int[] channels, int outputSize) {
        this.latentDim = latentDim;
        this.outputChannels = outputChannels;

        // Calculate projection size
        int projectionChannels = channels[channels.length - 1];
        int spatialSize = outputSize / (int) Math.pow(2, channels.length);
        int projectionSize = projectionChannels *
                           spatialSize * spatialSize * spatialSize;

        this.reshapeSize = new long[]{projectionChannels,
                                     spatialSize, spatialSize, spatialSize};

        // Project from latent to feature space
        this.fcProject = new Linear(latentDim, projectionSize);

        // Build transpose convolution blocks (reverse of encoder)
        this.deconvBlocks = buildDeconvBlocks(channels, outputChannels);

        // Register modules
        register_module("fc_project", fcProject);
        register_module("deconv_blocks", deconvBlocks);

        initializeWeights();
    }

    public Tensor forward(Tensor z) {
        // Project to feature space
        Tensor projected = fcProject.forward(z);

        // Reshape to 3D
        Tensor reshaped = projected.view(new long[]{
            z.size(0), reshapeSize[0], reshapeSize[1],
            reshapeSize[2], reshapeSize[3]
        });

        // Deconvolutional upsampling
        Tensor reconstruction = deconvBlocks.forward(reshaped);

        return reconstruction;
    }

    private Sequential buildDeconvBlocks(int[] channels, int outputChannels) {
        Sequential blocks = new Sequential();

        // Reverse channel order for decoding
        int[] reversedChannels = reverseArray(channels);

        for (int i = 0; i < reversedChannels.length; i++) {
            int inChannels = reversedChannels[i];
            int outChannels = (i == reversedChannels.length - 1) ?
                             outputChannels : reversedChannels[i + 1];

            blocks.add(createDeconvBlock(inChannels, outChannels,
                                        i < reversedChannels.length - 1));
        }

        // Final sigmoid activation for [0, 1] output
        blocks.add(Sigmoid());

        return blocks;
    }

    private Sequential createDeconvBlock(int in, int out, boolean upsample) {
        Sequential block = new Sequential();

        // TransposeConv3d + BatchNorm + LeakyReLU
        if (upsample) {
            block.add(ConvTranspose3d(in, out, new long[]{3, 3, 3},
                                     2, new long[]{1, 1, 1},
                                     new long[]{1, 1, 1})); // output_padding
            block.add(BatchNorm3d(out));
            block.add(LeakyReLU(0.2));
        } else {
            block.add(ConvTranspose3d(in, out, new long[]{3, 3, 3},
                                     1, new long[]{1, 1, 1},
                                     new long[]{0, 0, 0}));
        }

        return block;
    }

    private int[] reverseArray(int[] arr) {
        int[] reversed = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            reversed[i] = arr[arr.length - 1 - i];
        }
        return reversed;
    }

    private void initializeWeights() {
        // Kaiming normal initialization
    }
}
```

### 5.3 VariationalAutoencoder3D.java

**Responsibilities:**
- Main VAE model combining encoder and decoder
- Reparameterization trick
- Forward pass for training and inference

**Implementation:**
```java
package vtea.deeplearning.models;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class VariationalAutoencoder3D extends AbstractDeepLearningModel {
    private final VAEEncoder3D encoder;
    private final VAEDecoder3D decoder;
    private final int latentDim;
    private final VAEConfig config;

    public VariationalAutoencoder3D(VAEConfig config) {
        this.config = config;
        this.latentDim = config.getLatentDim();

        // Initialize encoder
        this.encoder = new VAEEncoder3D(
            config.getNumChannels(),
            latentDim,
            config.getEncoderChannels()
        );

        // Initialize decoder
        this.decoder = new VAEDecoder3D(
            latentDim,
            config.getNumChannels(),
            config.getEncoderChannels(),
            config.getInputSize()
        );

        // Register modules
        register_module("encoder", encoder);
        register_module("decoder", decoder);
    }

    @Override
    public VAEOutput forward(Tensor x) {
        // Encode
        VAEEncoder3D.EncoderOutput encoded = encoder.forward(x);
        Tensor mu = encoded.mu;
        Tensor logVar = encoded.logVar;

        // Reparameterization trick
        Tensor z = reparameterize(mu, logVar);

        // Decode
        Tensor reconstruction = decoder.forward(z);

        return new VAEOutput(reconstruction, mu, logVar, z);
    }

    /**
     * Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)
     */
    private Tensor reparameterize(Tensor mu, Tensor logVar) {
        if (!is_training()) {
            // During inference, just use mean
            return mu;
        }

        // σ = exp(0.5 * log(σ²))
        Tensor std = logVar.mul(0.5).exp();

        // ε ~ N(0, I)
        Tensor eps = randn_like(std);

        // z = μ + σ * ε
        return mu.add(std.mul(eps));
    }

    /**
     * Encode input to latent distribution
     */
    public VAEEncoder3D.EncoderOutput encode(Tensor x) {
        return encoder.forward(x);
    }

    /**
     * Decode latent vector to reconstruction
     */
    public Tensor decode(Tensor z) {
        return decoder.forward(z);
    }

    /**
     * Sample from latent space
     */
    public Tensor sample(int numSamples) {
        // Sample from prior N(0, I)
        Tensor z = randn(new long[]{numSamples, latentDim});

        // Decode to images
        return decoder.forward(z);
    }

    /**
     * Interpolate between two images in latent space
     */
    public Tensor[] interpolate(Tensor x1, Tensor x2, int steps) {
        // Encode both images
        Tensor mu1 = encoder.forward(x1).mu;
        Tensor mu2 = encoder.forward(x2).mu;

        Tensor[] interpolations = new Tensor[steps];

        for (int i = 0; i < steps; i++) {
            double alpha = (double) i / (steps - 1);

            // Linear interpolation in latent space
            Tensor z = mu1.mul(1 - alpha).add(mu2.mul(alpha));

            // Decode
            interpolations[i] = decoder.forward(z);
        }

        return interpolations;
    }

    public static class VAEOutput {
        public final Tensor reconstruction;
        public final Tensor mu;
        public final Tensor logVar;
        public final Tensor z;

        public VAEOutput(Tensor reconstruction, Tensor mu,
                        Tensor logVar, Tensor z) {
            this.reconstruction = reconstruction;
            this.mu = mu;
            this.logVar = logVar;
            this.z = z;
        }
    }
}
```

### 5.4 Model Variants

**BetaVAE3D.java:**
```java
public class BetaVAE3D extends VariationalAutoencoder3D {
    private final double beta;

    public BetaVAE3D(VAEConfig config, double beta) {
        super(config);
        this.beta = beta;
    }

    public double getBeta() {
        return beta;
    }

    // Beta is used in loss calculation (see Phase 2)
}
```

**ConditionalVAE3D.java:**
```java
public class ConditionalVAE3D extends VariationalAutoencoder3D {
    private final int numClasses;
    private final Embedding classEmbedding;

    public ConditionalVAE3D(VAEConfig config, int numClasses) {
        super(config);
        this.numClasses = numClasses;
        this.classEmbedding = new Embedding(numClasses, config.getLatentDim());
        register_module("class_embedding", classEmbedding);
    }

    public VAEOutput forward(Tensor x, Tensor classLabels) {
        // Encode
        VAEEncoder3D.EncoderOutput encoded = encoder.forward(x);

        // Concatenate class embedding with latent
        Tensor classEmbed = classEmbedding.forward(classLabels);
        Tensor mu = encoded.mu.add(classEmbed);

        // Rest of VAE forward pass
        // ...
    }
}
```

---

<a name="phase-2"></a>
## 6. Phase 2: Loss Functions & Training

### 6.1 VAELoss.java

**Responsibilities:**
- Combine reconstruction loss and KL divergence
- Support different reconstruction loss types
- Handle β-VAE weighting

**Implementation:**
```java
package vtea.deeplearning.loss;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class VAELoss {
    private final ReconstructionLoss reconstructionLoss;
    private final KLDivergenceLoss klLoss;
    private final double beta;
    private final boolean useKLWarmup;
    private final int warmupEpochs;
    private int currentEpoch;

    public VAELoss(ReconstructionType reconType, double beta,
                  boolean useKLWarmup, int warmupEpochs) {
        this.reconstructionLoss = new ReconstructionLoss(reconType);
        this.klLoss = new KLDivergenceLoss();
        this.beta = beta;
        this.useKLWarmup = useKLWarmup;
        this.warmupEpochs = warmupEpochs;
        this.currentEpoch = 0;
    }

    public LossOutput compute(Tensor reconstruction, Tensor target,
                             Tensor mu, Tensor logVar) {
        // Reconstruction loss
        Tensor reconLoss = reconstructionLoss.compute(reconstruction, target);

        // KL divergence
        Tensor kl = klLoss.compute(mu, logVar);

        // KL weighting (with optional warmup)
        double klWeight = getKLWeight();

        // Total loss = reconstruction + β * KL
        Tensor totalLoss = reconLoss.add(kl.mul(klWeight));

        return new LossOutput(totalLoss, reconLoss, kl, klWeight);
    }

    private double getKLWeight() {
        if (!useKLWarmup) {
            return beta;
        }

        // Linear warmup from 0 to beta
        if (currentEpoch < warmupEpochs) {
            return beta * (double) currentEpoch / warmupEpochs;
        }

        return beta;
    }

    public void setEpoch(int epoch) {
        this.currentEpoch = epoch;
    }

    public static class LossOutput {
        public final Tensor totalLoss;
        public final Tensor reconstructionLoss;
        public final Tensor klDivergence;
        public final double klWeight;

        public LossOutput(Tensor totalLoss, Tensor reconLoss,
                         Tensor kl, double klWeight) {
            this.totalLoss = totalLoss;
            this.reconstructionLoss = reconLoss;
            this.klDivergence = kl;
            this.klWeight = klWeight;
        }
    }
}
```

### 6.2 ReconstructionLoss.java

```java
package vtea.deeplearning.loss;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public enum ReconstructionType {
    MSE,        // Mean Squared Error (for continuous data)
    BCE,        // Binary Cross-Entropy (for normalized [0,1] data)
    L1          // L1 loss (for robustness)
}

public class ReconstructionLoss {
    private final ReconstructionType type;

    public ReconstructionLoss(ReconstructionType type) {
        this.type = type;
    }

    public Tensor compute(Tensor reconstruction, Tensor target) {
        switch (type) {
            case MSE:
                return mse_loss(reconstruction, target, Reduction.Mean);

            case BCE:
                return binary_cross_entropy(reconstruction, target,
                                           null, Reduction.Mean);

            case L1:
                return l1_loss(reconstruction, target, Reduction.Mean);

            default:
                throw new IllegalArgumentException("Unknown loss type: " + type);
        }
    }
}
```

### 6.3 KLDivergenceLoss.java

```java
package vtea.deeplearning.loss;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class KLDivergenceLoss {

    /**
     * Analytical KL divergence for Gaussian distributions:
     * KL(N(μ, σ²) || N(0, I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
     */
    public Tensor compute(Tensor mu, Tensor logVar) {
        // KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        Tensor kl = logVar.add(1.0)
                         .sub(mu.pow(2.0))
                         .sub(logVar.exp());

        // Sum over latent dimensions, mean over batch
        kl = kl.sum(new long[]{1}).mul(-0.5).mean();

        return kl;
    }

    /**
     * Per-sample KL divergence (for analysis)
     */
    public Tensor computePerSample(Tensor mu, Tensor logVar) {
        Tensor kl = logVar.add(1.0)
                         .sub(mu.pow(2.0))
                         .sub(logVar.exp());

        // Sum over latent dimensions only
        return kl.sum(new long[]{1}).mul(-0.5);
    }
}
```

### 6.4 VAETrainer.java

**Responsibilities:**
- Training loop for VAE
- Track reconstruction, KL, and ELBO metrics
- Early stopping based on validation loss

**Implementation:**
```java
package vtea.deeplearning.training;

import vtea.deeplearning.models.*;
import vtea.deeplearning.loss.*;
import org.bytedeco.pytorch.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VAETrainer {
    private static final Logger logger = LoggerFactory.getLogger(VAETrainer.class);

    private final VariationalAutoencoder3D model;
    private final VAELoss lossFunction;
    private final Optimizer optimizer;
    private final VAEMetrics metrics;
    private final ModelCheckpoint checkpoint;
    private final int maxEpochs;

    public VAETrainer(VariationalAutoencoder3D model, VAEConfig config,
                     String checkpointDir) {
        this.model = model;
        this.lossFunction = new VAELoss(
            config.getReconstructionType(),
            config.getBeta(),
            config.isWarmupKL(),
            config.getWarmupEpochs()
        );
        this.optimizer = createOptimizer(config);
        this.metrics = new VAEMetrics();
        this.checkpoint = new ModelCheckpoint(checkpointDir);
        this.maxEpochs = config.getEpochs();
    }

    public TrainingResult train(DataLoader trainLoader,
                               DataLoader valLoader) {
        logger.info("Starting VAE training for {} epochs", maxEpochs);

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            lossFunction.setEpoch(epoch);

            // Training phase
            EpochMetrics trainMetrics = trainEpoch(trainLoader, epoch);

            // Validation phase
            EpochMetrics valMetrics = validateEpoch(valLoader, epoch);

            // Log metrics
            logEpochMetrics(epoch, trainMetrics, valMetrics);

            // Save checkpoint if best model
            if (valMetrics.totalLoss < metrics.getBestValLoss()) {
                checkpoint.save(model, epoch, valMetrics.totalLoss);
                logger.info("New best model saved (val loss: {:.4f})",
                          valMetrics.totalLoss);
            }

            // Early stopping check
            if (metrics.shouldEarlyStop(valMetrics.totalLoss)) {
                logger.info("Early stopping at epoch {}", epoch);
                break;
            }
        }

        return new TrainingResult(metrics);
    }

    private EpochMetrics trainEpoch(DataLoader loader, int epoch) {
        model.train();
        metrics.resetEpoch();

        int batchIdx = 0;
        while (loader.hasNext()) {
            Batch batch = loader.next();
            Tensor input = batch.getData();

            // Forward pass
            VariationalAutoencoder3D.VAEOutput output = model.forward(input);

            // Compute loss
            VAELoss.LossOutput loss = lossFunction.compute(
                output.reconstruction,
                input,  // Target is input for autoencoder
                output.mu,
                output.logVar
            );

            // Backward pass
            optimizer.zero_grad();
            loss.totalLoss.backward();
            optimizer.step();

            // Track metrics
            metrics.update(
                loss.totalLoss.item().getDouble(),
                loss.reconstructionLoss.item().getDouble(),
                loss.klDivergence.item().getDouble()
            );

            // Periodic logging
            if (batchIdx % 10 == 0) {
                logger.debug("Epoch {} [{}/{}] Loss: {:.4f} (Recon: {:.4f}, KL: {:.4f})",
                           epoch, batchIdx, loader.size(),
                           loss.totalLoss.item().getDouble(),
                           loss.reconstructionLoss.item().getDouble(),
                           loss.klDivergence.item().getDouble());
            }

            batchIdx++;
        }

        return metrics.getEpochMetrics();
    }

    private EpochMetrics validateEpoch(DataLoader loader, int epoch) {
        model.eval();
        VAEMetrics valMetrics = new VAEMetrics();

        no_grad(() -> {
            while (loader.hasNext()) {
                Batch batch = loader.next();
                Tensor input = batch.getData();

                // Forward pass
                VariationalAutoencoder3D.VAEOutput output = model.forward(input);

                // Compute loss
                VAELoss.LossOutput loss = lossFunction.compute(
                    output.reconstruction,
                    input,
                    output.mu,
                    output.logVar
                );

                // Track metrics
                valMetrics.update(
                    loss.totalLoss.item().getDouble(),
                    loss.reconstructionLoss.item().getDouble(),
                    loss.klDivergence.item().getDouble()
                );
            }
        });

        return valMetrics.getEpochMetrics();
    }

    private Optimizer createOptimizer(VAEConfig config) {
        // Adam optimizer
        return new Adam(model.parameters(), config.getLearningRate());
    }

    private void logEpochMetrics(int epoch, EpochMetrics train,
                                EpochMetrics val) {
        logger.info("Epoch {}/{}: Train Loss={:.4f} (Recon={:.4f}, KL={:.4f}), " +
                   "Val Loss={:.4f} (Recon={:.4f}, KL={:.4f})",
                   epoch + 1, maxEpochs,
                   train.totalLoss, train.reconLoss, train.klLoss,
                   val.totalLoss, val.reconLoss, val.klLoss);
    }
}
```

### 6.5 VAEMetrics.java

```java
package vtea.deeplearning.training;

import java.util.ArrayList;
import java.util.List;

public class VAEMetrics {
    private List<Double> totalLosses = new ArrayList<>();
    private List<Double> reconLosses = new ArrayList<>();
    private List<Double> klLosses = new ArrayList<>();

    private double epochTotalLoss = 0.0;
    private double epochReconLoss = 0.0;
    private double epochKLLoss = 0.0;
    private int batchCount = 0;

    private double bestValLoss = Double.MAX_VALUE;
    private int patience = 10;
    private int patienceCounter = 0;

    public void update(double totalLoss, double reconLoss, double klLoss) {
        epochTotalLoss += totalLoss;
        epochReconLoss += reconLoss;
        epochKLLoss += klLoss;
        batchCount++;
    }

    public void resetEpoch() {
        epochTotalLoss = 0.0;
        epochReconLoss = 0.0;
        epochKLLoss = 0.0;
        batchCount = 0;
    }

    public EpochMetrics getEpochMetrics() {
        double avgTotal = epochTotalLoss / batchCount;
        double avgRecon = epochReconLoss / batchCount;
        double avgKL = epochKLLoss / batchCount;

        totalLosses.add(avgTotal);
        reconLosses.add(avgRecon);
        klLosses.add(avgKL);

        return new EpochMetrics(avgTotal, avgRecon, avgKL);
    }

    public boolean shouldEarlyStop(double valLoss) {
        if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            patienceCounter = 0;
            return false;
        }

        patienceCounter++;
        return patienceCounter >= patience;
    }

    public double getBestValLoss() {
        return bestValLoss;
    }

    public static class EpochMetrics {
        public final double totalLoss;
        public final double reconLoss;
        public final double klLoss;

        public EpochMetrics(double total, double recon, double kl) {
            this.totalLoss = total;
            this.reconLoss = recon;
            this.klLoss = kl;
        }
    }
}
```

---

<a name="phase-3"></a>
## 7. Phase 3: Latent Space Analysis

### 7.1 VAELatentExtractor.java

**Responsibilities:**
- Extract latent representations for all cells
- Store latent vectors with MicroObject associations
- Compute uncertainty estimates

**Implementation:**
```java
package vtea.deeplearning.inference;

import vtea.deeplearning.models.VariationalAutoencoder3D;
import vtea.deeplearning.data.CellRegionExtractor;
import vteaobjects.MicroObject;
import org.bytedeco.pytorch.*;
import java.util.*;

public class VAELatentExtractor {
    private final VariationalAutoencoder3D model;
    private final CellRegionExtractor regionExtractor;
    private final int numSamples; // For uncertainty estimation

    public VAELatentExtractor(VariationalAutoencoder3D model,
                             CellRegionExtractor regionExtractor,
                             int numSamples) {
        this.model = model;
        this.regionExtractor = regionExtractor;
        this.numSamples = numSamples;
    }

    /**
     * Extract latent representations for all cells
     */
    public Map<MicroObject, LatentRepresentation> extractLatents(
            List<MicroObject> cells,
            ImageStack[] imageStacks) {

        Map<MicroObject, LatentRepresentation> latents = new HashMap<>();
        model.eval();

        no_grad(() -> {
            for (MicroObject cell : cells) {
                // Extract 3D region around cell
                Tensor region = regionExtractor.extractRegion(cell, imageStacks);

                // Encode to latent distribution
                VAEEncoder3D.EncoderOutput encoded = model.encode(region);

                // Extract mean (deterministic representation)
                float[] muArray = tensorToFloatArray(encoded.mu);

                // Compute uncertainty (standard deviation)
                float[] stdArray = logVarToStd(encoded.logVar);

                // Sample multiple times for uncertainty estimation
                float[][] samples = null;
                if (numSamples > 1) {
                    samples = sampleLatent(encoded.mu, encoded.logVar, numSamples);
                }

                LatentRepresentation latent = new LatentRepresentation(
                    muArray, stdArray, samples
                );

                latents.put(cell, latent);
            }
        });

        return latents;
    }

    /**
     * Sample from latent distribution multiple times
     */
    private float[][] sampleLatent(Tensor mu, Tensor logVar, int numSamples) {
        float[][] samples = new float[numSamples][];

        for (int i = 0; i < numSamples; i++) {
            Tensor std = logVar.mul(0.5).exp();
            Tensor eps = randn_like(std);
            Tensor z = mu.add(std.mul(eps));
            samples[i] = tensorToFloatArray(z);
        }

        return samples;
    }

    private float[] tensorToFloatArray(Tensor t) {
        // Convert PyTorch tensor to float array
        // Implementation depends on bytedeco API
    }

    private float[] logVarToStd(Tensor logVar) {
        Tensor std = logVar.mul(0.5).exp();
        return tensorToFloatArray(std);
    }

    /**
     * Container for latent representation
     */
    public static class LatentRepresentation {
        public final float[] mean;           // Point estimate
        public final float[] std;            // Uncertainty
        public final float[][] samples;      // Multiple samples (optional)

        public LatentRepresentation(float[] mean, float[] std,
                                   float[][] samples) {
            this.mean = mean;
            this.std = std;
            this.samples = samples;
        }

        /**
         * Get average uncertainty across dimensions
         */
        public double getAverageUncertainty() {
            double sum = 0;
            for (float s : std) {
                sum += s;
            }
            return sum / std.length;
        }
    }
}
```

### 7.2 LatentSpaceVisualizer.java

**Responsibilities:**
- 2D/3D visualization of latent space
- Apply dimensionality reduction (t-SNE, UMAP) if latent > 3D
- Color by class, cluster, or quality metrics

**Implementation:**
```java
package vtea.deeplearning.analysis;

import vtea.featureprocessing.tSNE;
import vteaobjects.MicroObject;
import org.jfree.chart.*;
import java.util.*;

public class LatentSpaceVisualizer {

    /**
     * Create 2D projection of latent space
     */
    public JFreeChart create2DProjection(
            Map<MicroObject, VAELatentExtractor.LatentRepresentation> latents,
            ProjectionMethod method,
            ColorScheme colorScheme) {

        // Convert latent representations to matrix
        double[][] latentMatrix = toMatrix(latents);

        // Apply dimensionality reduction if needed
        double[][] projection;
        if (latentMatrix[0].length > 2) {
            projection = reduce2D(latentMatrix, method);
        } else {
            projection = latentMatrix;
        }

        // Create scatter plot
        return createScatterPlot(projection, latents, colorScheme);
    }

    /**
     * Create interactive 3D visualization
     */
    public void create3DVisualization(
            Map<MicroObject, VAELatentExtractor.LatentRepresentation> latents,
            ProjectionMethod method) {

        double[][] latentMatrix = toMatrix(latents);

        // Reduce to 3D if necessary
        double[][] projection;
        if (latentMatrix[0].length > 3) {
            projection = reduce3D(latentMatrix, method);
        } else {
            projection = latentMatrix;
        }

        // Create 3D visualization (integrate with MicroExplorer)
        // Use existing VTEA 3D plotting infrastructure
    }

    private double[][] reduce2D(double[][] data, ProjectionMethod method) {
        switch (method) {
            case TSNE:
                return tSNE.run(data, 2);
            case UMAP:
                return runUMAP(data, 2);
            case PCA:
                return runPCA(data, 2);
            default:
                throw new IllegalArgumentException("Unknown method: " + method);
        }
    }

    private double[][] toMatrix(
            Map<MicroObject, VAELatentExtractor.LatentRepresentation> latents) {
        int n = latents.size();
        int d = latents.values().iterator().next().mean.length;

        double[][] matrix = new double[n][d];
        int i = 0;
        for (VAELatentExtractor.LatentRepresentation latent : latents.values()) {
            for (int j = 0; j < d; j++) {
                matrix[i][j] = latent.mean[j];
            }
            i++;
        }

        return matrix;
    }

    public enum ProjectionMethod {
        NONE,      // Use raw latent space (if 2D/3D)
        TSNE,      // t-SNE
        UMAP,      // UMAP (requires additional dependency)
        PCA        // PCA
    }

    public enum ColorScheme {
        CLASS,           // Color by manual class
        CLUSTER,         // Color by cluster
        UNCERTAINTY,     // Color by latent uncertainty
        RECON_ERROR,     // Color by reconstruction error
        CUSTOM           // Custom coloring
    }
}
```

### 7.3 LatentSpaceInterpolator.java

```java
package vtea.deeplearning.analysis;

import vtea.deeplearning.models.VariationalAutoencoder3D;
import org.bytedeco.pytorch.*;

public class LatentSpaceInterpolator {
    private final VariationalAutoencoder3D model;

    public LatentSpaceInterpolator(VariationalAutoencoder3D model) {
        this.model = model;
    }

    /**
     * Linear interpolation between two cells
     */
    public List<ImageStack> interpolate(Tensor cell1, Tensor cell2, int steps) {
        model.eval();
        List<ImageStack> interpolations = new ArrayList<>();

        no_grad(() -> {
            // Encode both cells
            Tensor z1 = model.encode(cell1).mu;
            Tensor z2 = model.encode(cell2).mu;

            for (int i = 0; i < steps; i++) {
                double alpha = (double) i / (steps - 1);

                // Linear interpolation
                Tensor z = z1.mul(1 - alpha).add(z2.mul(alpha));

                // Decode
                Tensor reconstruction = model.decode(z);

                // Convert to ImageStack
                ImageStack stack = tensorToImageStack(reconstruction);
                interpolations.add(stack);
            }
        });

        return interpolations;
    }

    /**
     * Spherical interpolation (better for high dimensions)
     */
    public List<ImageStack> sphericalInterpolate(Tensor cell1, Tensor cell2,
                                                 int steps) {
        // Slerp interpolation
        // More appropriate for latent spaces
    }
}
```

---

<a name="phase-4"></a>
## 8. Phase 4: Reconstruction & Quality Metrics

### 8.1 VAEReconstructor.java

**Responsibilities:**
- Generate reconstructions for cells
- Compute reconstruction quality metrics
- Identify poor quality segmentations

**Implementation:**
```java
package vtea.deeplearning.inference;

import vtea.deeplearning.models.VariationalAutoencoder3D;
import vteaobjects.MicroObject;
import org.bytedeco.pytorch.*;
import java.util.*;

public class VAEReconstructor {
    private final VariationalAutoencoder3D model;
    private final CellRegionExtractor regionExtractor;

    public VAEReconstructor(VariationalAutoencoder3D model,
                           CellRegionExtractor regionExtractor) {
        this.model = model;
        this.regionExtractor = regionExtractor;
    }

    /**
     * Reconstruct all cells and compute quality metrics
     */
    public Map<MicroObject, ReconstructionResult> reconstructAll(
            List<MicroObject> cells,
            ImageStack[] imageStacks) {

        Map<MicroObject, ReconstructionResult> results = new HashMap<>();
        model.eval();

        no_grad(() -> {
            for (MicroObject cell : cells) {
                // Extract original region
                Tensor original = regionExtractor.extractRegion(cell, imageStacks);

                // Forward pass
                VariationalAutoencoder3D.VAEOutput output = model.forward(original);

                // Compute metrics
                double mse = computeMSE(original, output.reconstruction);
                double ssim = computeSSIM(original, output.reconstruction);
                double psnr = computePSNR(original, output.reconstruction);

                // Convert reconstruction to ImageStack
                ImageStack reconstruction = tensorToImageStack(output.reconstruction);

                ReconstructionResult result = new ReconstructionResult(
                    reconstruction,
                    mse,
                    ssim,
                    psnr,
                    tensorToFloatArray(output.z)
                );

                results.put(cell, result);
            }
        });

        return results;
    }

    /**
     * Identify low-quality cells based on reconstruction error
     */
    public List<MicroObject> identifyLowQualityCells(
            Map<MicroObject, ReconstructionResult> results,
            double mseThreshold) {

        List<MicroObject> lowQuality = new ArrayList<>();

        for (Map.Entry<MicroObject, ReconstructionResult> entry : results.entrySet()) {
            if (entry.getValue().mse > mseThreshold) {
                lowQuality.add(entry.getKey());
            }
        }

        return lowQuality;
    }

    /**
     * Compute Mean Squared Error
     */
    private double computeMSE(Tensor original, Tensor reconstruction) {
        Tensor diff = original.sub(reconstruction);
        Tensor squared = diff.pow(2.0);
        return squared.mean().item().getDouble();
    }

    /**
     * Compute Structural Similarity Index (SSIM)
     */
    private double computeSSIM(Tensor original, Tensor reconstruction) {
        // SSIM implementation for 3D volumes
        // Can use existing image quality libraries or implement
        return 0.0; // Placeholder
    }

    /**
     * Compute Peak Signal-to-Noise Ratio (PSNR)
     */
    private double computePSNR(Tensor original, Tensor reconstruction) {
        double mse = computeMSE(original, reconstruction);
        if (mse == 0) return Double.MAX_VALUE;

        double maxPixel = 1.0; // Assuming normalized [0, 1]
        return 20 * Math.log10(maxPixel / Math.sqrt(mse));
    }

    public static class ReconstructionResult {
        public final ImageStack reconstruction;
        public final double mse;
        public final double ssim;
        public final double psnr;
        public final float[] latent;

        public ReconstructionResult(ImageStack recon, double mse,
                                   double ssim, double psnr, float[] latent) {
            this.reconstruction = recon;
            this.mse = mse;
            this.ssim = ssim;
            this.psnr = psnr;
            this.latent = latent;
        }

        public double getQualityScore() {
            // Combined quality score (lower is worse)
            // Weighted combination of metrics
            return mse * 0.5 + (1 - ssim) * 0.5;
        }
    }
}
```

### 8.2 ReconstructionQualityAnalyzer.java

```java
package vtea.deeplearning.analysis;

import vtea.deeplearning.inference.VAEReconstructor.ReconstructionResult;
import vteaobjects.MicroObject;
import org.jfree.chart.*;
import java.util.*;

public class ReconstructionQualityAnalyzer {

    /**
     * Generate quality distribution plots
     */
    public JFreeChart plotQualityDistribution(
            Map<MicroObject, ReconstructionResult> results) {

        // Create histogram of MSE values
        double[] mseValues = results.values().stream()
            .mapToDouble(r -> r.mse)
            .toArray();

        return createHistogram(mseValues, "MSE Distribution", "MSE", "Count");
    }

    /**
     * Identify outliers using statistical methods
     */
    public List<MicroObject> identifyOutliers(
            Map<MicroObject, ReconstructionResult> results,
            OutlierMethod method) {

        double[] qualityScores = results.values().stream()
            .mapToDouble(r -> r.getQualityScore())
            .toArray();

        double threshold = computeOutlierThreshold(qualityScores, method);

        List<MicroObject> outliers = new ArrayList<>();
        for (Map.Entry<MicroObject, ReconstructionResult> entry : results.entrySet()) {
            if (entry.getValue().getQualityScore() > threshold) {
                outliers.add(entry.getKey());
            }
        }

        return outliers;
    }

    private double computeOutlierThreshold(double[] scores, OutlierMethod method) {
        switch (method) {
            case IQR:
                return computeIQRThreshold(scores);
            case PERCENTILE:
                return computePercentileThreshold(scores, 95.0);
            case MAD:
                return computeMADThreshold(scores);
            default:
                throw new IllegalArgumentException("Unknown method: " + method);
        }
    }

    private double computeIQRThreshold(double[] scores) {
        Arrays.sort(scores);
        int n = scores.length;
        double q1 = scores[n / 4];
        double q3 = scores[3 * n / 4];
        double iqr = q3 - q1;
        return q3 + 1.5 * iqr;
    }

    public enum OutlierMethod {
        IQR,          // Interquartile range
        PERCENTILE,   // Top percentile
        MAD           // Median absolute deviation
    }
}
```

---

<a name="phase-5"></a>
## 9. Phase 5: Dimensionality Reduction Integration

### 9.1 VAEFeatureExtraction.java

**Responsibilities:**
- VTEA FeatureProcessing plugin
- Extract latent features for downstream analysis
- Replace traditional feature extraction with learned representations

**Implementation:**
```java
package vtea.featureprocessing;

import vtea.deeplearning.models.VariationalAutoencoder3D;
import vtea.deeplearning.inference.VAELatentExtractor;
import vteaobjects.MicroObject;
import org.scijava.plugin.Plugin;
import java.util.*;

@Plugin(type = FeatureProcessing.class)
public class VAEFeatureExtraction extends AbstractFeatureProcessing {

    private VariationalAutoencoder3D model;
    private VAELatentExtractor latentExtractor;
    private Map<MicroObject, float[]> latentFeatures;

    public VAEFeatureExtraction() {
        super();
        VERSION = "1.0";
        AUTHOR = "VTEA Development Team";
        COMMENT = "Extract learned features using 3D VAE";
        NAME = "VAE Feature Extraction";
        KEY = "VAEFeatures";
    }

    @Override
    public void processObjects(ArrayList<MicroObject> objects,
                               ImageStack[] imageStacks) {

        // Load trained VAE model
        loadModel();

        // Extract latent representations
        Map<MicroObject, VAELatentExtractor.LatentRepresentation> latents =
            latentExtractor.extractLatents(objects, imageStacks);

        // Store latent vectors as features
        latentFeatures = new HashMap<>();
        for (Map.Entry<MicroObject, VAELatentExtractor.LatentRepresentation> entry
             : latents.entrySet()) {
            latentFeatures.put(entry.getKey(), entry.getValue().mean);
        }

        // Add features to MicroObjects
        addFeaturesToObjects(objects);
    }

    private void addFeaturesToObjects(ArrayList<MicroObject> objects) {
        for (MicroObject obj : objects) {
            float[] latent = latentFeatures.get(obj);

            // Add each latent dimension as a feature
            ArrayList<Number> features = new ArrayList<>();
            for (float value : latent) {
                features.add(value);
            }

            obj.getFeatures().add(features);
        }
    }

    @Override
    public ArrayList<String> getFeatureNames() {
        ArrayList<String> names = new ArrayList<>();
        int latentDim = model.getLatentDim();

        for (int i = 0; i < latentDim; i++) {
            names.add("VAE_Latent_" + i);
        }

        return names;
    }

    private void loadModel() {
        // Load pre-trained model from checkpoint
        String modelPath = getModelPath();
        model = ModelCheckpoint.load(modelPath);
        latentExtractor = new VAELatentExtractor(model,
            new CellRegionExtractor(), 1);
    }

    @Override
    public JPanel getPanel() {
        // UI panel for model selection and parameters
        return new VAEFeatureExtractionPanel(this);
    }
}
```

### 9.2 VAEClustering.java

```java
package vtea.featureprocessing;

import vtea.deeplearning.inference.VAELatentExtractor;
import smile.clustering.KMeans;
import org.scijava.plugin.Plugin;

@Plugin(type = FeatureProcessing.class)
public class VAEClustering extends AbstractFeatureProcessing {

    private int numClusters = 5;
    private VAELatentExtractor latentExtractor;

    public VAEClustering() {
        super();
        VERSION = "1.0";
        NAME = "VAE K-Means Clustering";
        KEY = "VAEClustering";
        COMMENT = "Cluster cells in VAE latent space";
    }

    @Override
    public void processObjects(ArrayList<MicroObject> objects,
                               ImageStack[] imageStacks) {

        // Extract latent representations
        Map<MicroObject, VAELatentExtractor.LatentRepresentation> latents =
            latentExtractor.extractLatents(objects, imageStacks);

        // Convert to matrix for clustering
        double[][] latentMatrix = toMatrix(latents);

        // Run K-Means in latent space
        KMeans kmeans = new KMeans(latentMatrix, numClusters);
        int[] clusterAssignments = kmeans.getClusterLabel();

        // Assign clusters to objects
        int i = 0;
        for (MicroObject obj : objects) {
            obj.setCluster(clusterAssignments[i]);
            i++;
        }
    }

    @Override
    public JPanel getPanel() {
        return new VAEClusteringPanel(this);
    }
}
```

---

<a name="phase-6"></a>
## 10. Phase 6: Anomaly Detection

### 10.1 VAEAnomalyDetector.java

**Responsibilities:**
- Detect anomalous cells based on reconstruction error
- Compute anomaly scores
- Integration with VTEA quality control workflow

**Implementation:**
```java
package vtea.deeplearning.inference;

import vtea.deeplearning.models.VariationalAutoencoder3D;
import vteaobjects.MicroObject;
import org.bytedeco.pytorch.*;
import java.util.*;

public class VAEAnomalyDetector {
    private final VariationalAutoencoder3D model;
    private final CellRegionExtractor regionExtractor;
    private double threshold;

    public VAEAnomalyDetector(VariationalAutoencoder3D model,
                             CellRegionExtractor regionExtractor,
                             double threshold) {
        this.model = model;
        this.regionExtractor = regionExtractor;
        this.threshold = threshold;
    }

    /**
     * Compute anomaly scores for all cells
     */
    public Map<MicroObject, AnomalyScore> computeAnomalyScores(
            List<MicroObject> cells,
            ImageStack[] imageStacks) {

        Map<MicroObject, AnomalyScore> scores = new HashMap<>();
        model.eval();

        no_grad(() -> {
            for (MicroObject cell : cells) {
                // Extract region
                Tensor region = regionExtractor.extractRegion(cell, imageStacks);

                // Forward pass
                VariationalAutoencoder3D.VAEOutput output = model.forward(region);

                // Compute reconstruction error
                double reconError = computeReconstructionError(
                    region, output.reconstruction
                );

                // Compute KL divergence (distance from prior)
                double kl = computeKLDivergence(output.mu, output.logVar);

                // Combined anomaly score
                double anomalyScore = reconError + kl;

                AnomalyScore score = new AnomalyScore(
                    reconError,
                    kl,
                    anomalyScore,
                    anomalyScore > threshold
                );

                scores.put(cell, score);
            }
        });

        return scores;
    }

    /**
     * Automatically determine threshold from data
     */
    public double computeThreshold(Map<MicroObject, AnomalyScore> scores,
                                   double percentile) {
        double[] anomalyScores = scores.values().stream()
            .mapToDouble(s -> s.totalScore)
            .sorted()
            .toArray();

        int index = (int) (anomalyScores.length * percentile / 100.0);
        return anomalyScores[index];
    }

    /**
     * Filter out anomalous cells
     */
    public List<MicroObject> filterAnomalies(
            Map<MicroObject, AnomalyScore> scores) {

        List<MicroObject> filtered = new ArrayList<>();

        for (Map.Entry<MicroObject, AnomalyScore> entry : scores.entrySet()) {
            if (!entry.getValue().isAnomaly) {
                filtered.add(entry.getKey());
            }
        }

        return filtered;
    }

    private double computeReconstructionError(Tensor original,
                                             Tensor reconstruction) {
        Tensor diff = original.sub(reconstruction);
        return diff.pow(2.0).mean().item().getDouble();
    }

    private double computeKLDivergence(Tensor mu, Tensor logVar) {
        Tensor kl = logVar.add(1.0)
                         .sub(mu.pow(2.0))
                         .sub(logVar.exp());
        return kl.sum().mul(-0.5).item().getDouble();
    }

    public static class AnomalyScore {
        public final double reconstructionError;
        public final double klDivergence;
        public final double totalScore;
        public final boolean isAnomaly;

        public AnomalyScore(double reconError, double kl,
                           double total, boolean isAnomaly) {
            this.reconstructionError = reconError;
            this.klDivergence = kl;
            this.totalScore = total;
            this.isAnomaly = isAnomaly;
        }
    }
}
```

### 10.2 VAEAnomalyDetection.java (Plugin)

```java
package vtea.featureprocessing;

import vtea.deeplearning.inference.VAEAnomalyDetector;
import vteaobjects.MicroObject;
import org.scijava.plugin.Plugin;
import java.util.*;

@Plugin(type = FeatureProcessing.class)
public class VAEAnomalyDetection extends AbstractFeatureProcessing {

    private VAEAnomalyDetector detector;
    private double percentileThreshold = 95.0;

    public VAEAnomalyDetection() {
        super();
        VERSION = "1.0";
        NAME = "VAE Anomaly Detection";
        KEY = "VAEAnomaly";
        COMMENT = "Detect anomalous cells using VAE reconstruction";
    }

    @Override
    public void processObjects(ArrayList<MicroObject> objects,
                               ImageStack[] imageStacks) {

        // Compute anomaly scores
        Map<MicroObject, VAEAnomalyDetector.AnomalyScore> scores =
            detector.computeAnomalyScores(objects, imageStacks);

        // Auto-determine threshold
        double threshold = detector.computeThreshold(scores, percentileThreshold);

        // Mark anomalies
        for (Map.Entry<MicroObject, VAEAnomalyDetector.AnomalyScore> entry
             : scores.entrySet()) {
            MicroObject obj = entry.getKey();
            VAEAnomalyDetector.AnomalyScore score = entry.getValue();

            // Add anomaly score as feature
            ArrayList<Number> features = new ArrayList<>();
            features.add(score.totalScore);
            features.add(score.isAnomaly ? 1 : 0);
            obj.getFeatures().add(features);

            // Optionally mark for exclusion
            if (score.isAnomaly) {
                obj.setExcluded(true);
            }
        }
    }

    @Override
    public ArrayList<String> getFeatureNames() {
        ArrayList<String> names = new ArrayList<>();
        names.add("VAE_Anomaly_Score");
        names.add("VAE_Is_Anomaly");
        return names;
    }

    @Override
    public JPanel getPanel() {
        return new VAEAnomalyDetectionPanel(this);
    }
}
```

---

<a name="phase-7"></a>
## 11. Phase 7: UI Components

### 11.1 VAETrainingPanel.java

**Responsibilities:**
- UI for training VAE models
- Configure architecture, hyperparameters
- Monitor training progress
- Load/save models

**Implementation:**
```java
package vtea.exploration.plottools.panels;

import vtea.deeplearning.models.*;
import vtea.deeplearning.training.*;
import javax.swing.*;
import java.awt.*;

public class VAETrainingPanel extends JPanel {

    private JComboBox<String> architectureCombo;
    private JSpinner latentDimSpinner;
    private JSpinner epochsSpinner;
    private JSpinner batchSizeSpinner;
    private JTextField learningRateField;
    private JSpinner betaSpinner;
    private JCheckBox warmupCheckBox;
    private JProgressBar trainingProgress;
    private JTextArea logArea;
    private JButton trainButton;
    private JButton stopButton;

    private VAETrainer trainer;

    public VAETrainingPanel() {
        setLayout(new BorderLayout());

        // Configuration panel
        JPanel configPanel = createConfigPanel();
        add(configPanel, BorderLayout.NORTH);

        // Progress panel
        JPanel progressPanel = createProgressPanel();
        add(progressPanel, BorderLayout.CENTER);

        // Control buttons
        JPanel controlPanel = createControlPanel();
        add(controlPanel, BorderLayout.SOUTH);
    }

    private JPanel createConfigPanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(BorderFactory.createTitledBorder("VAE Configuration"));

        // Architecture selection
        panel.add(new JLabel("Architecture:"));
        architectureCombo = new JComboBox<>(new String[]{
            "Small (32³, 16D)", "Medium (64³, 32D)", "Large (128³, 64D)"
        });
        panel.add(architectureCombo);

        // Latent dimensions
        panel.add(new JLabel("Latent Dimensions:"));
        latentDimSpinner = new JSpinner(new SpinnerNumberModel(32, 8, 256, 8));
        panel.add(latentDimSpinner);

        // Training parameters
        panel.add(new JLabel("Epochs:"));
        epochsSpinner = new JSpinner(new SpinnerNumberModel(100, 1, 1000, 10));
        panel.add(epochsSpinner);

        panel.add(new JLabel("Batch Size:"));
        batchSizeSpinner = new JSpinner(new SpinnerNumberModel(16, 1, 128, 4));
        panel.add(batchSizeSpinner);

        panel.add(new JLabel("Learning Rate:"));
        learningRateField = new JTextField("1e-4");
        panel.add(learningRateField);

        // Beta-VAE parameter
        panel.add(new JLabel("Beta (KL weight):"));
        betaSpinner = new JSpinner(new SpinnerNumberModel(1.0, 0.1, 10.0, 0.1));
        panel.add(betaSpinner);

        panel.add(new JLabel("KL Warmup:"));
        warmupCheckBox = new JCheckBox("Enable");
        warmupCheckBox.setSelected(true);
        panel.add(warmupCheckBox);

        return panel;
    }

    private JPanel createProgressPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Training Progress"));

        trainingProgress = new JProgressBar(0, 100);
        trainingProgress.setStringPainted(true);
        panel.add(trainingProgress, BorderLayout.NORTH);

        logArea = new JTextArea(10, 40);
        logArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(logArea);
        panel.add(scrollPane, BorderLayout.CENTER);

        return panel;
    }

    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new FlowLayout());

        trainButton = new JButton("Start Training");
        trainButton.addActionListener(e -> startTraining());
        panel.add(trainButton);

        stopButton = new JButton("Stop");
        stopButton.setEnabled(false);
        stopButton.addActionListener(e -> stopTraining());
        panel.add(stopButton);

        JButton loadButton = new JButton("Load Model");
        loadButton.addActionListener(e -> loadModel());
        panel.add(loadButton);

        JButton saveButton = new JButton("Save Model");
        saveButton.addActionListener(e -> saveModel());
        panel.add(saveButton);

        return panel;
    }

    private void startTraining() {
        // Create VAE config from UI
        VAEConfig config = buildConfig();

        // Create model and trainer
        VariationalAutoencoder3D model = new VariationalAutoencoder3D(config);
        trainer = new VAETrainer(model, config, "./checkpoints");

        // Run training in background thread
        SwingWorker<Void, String> worker = new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                // Training loop with progress updates
                // ...
                return null;
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                for (String log : chunks) {
                    logArea.append(log + "\n");
                }
            }
        };

        worker.execute();
        trainButton.setEnabled(false);
        stopButton.setEnabled(true);
    }

    private VAEConfig buildConfig() {
        VAEConfig config = new VAEConfig();
        config.setLatentDim((Integer) latentDimSpinner.getValue());
        config.setEpochs((Integer) epochsSpinner.getValue());
        config.setBatchSize((Integer) batchSizeSpinner.getValue());
        config.setLearningRate(Double.parseDouble(learningRateField.getText()));
        config.setBeta((Double) betaSpinner.getValue());
        config.setWarmupKL(warmupCheckBox.isSelected());
        return config;
    }

    private void stopTraining() {
        // Stop training
    }

    private void loadModel() {
        // Load pre-trained model
    }

    private void saveModel() {
        // Save current model
    }
}
```

### 11.2 VAELatentSpacePanel.java

```java
package vtea.exploration.plottools.panels;

import vtea.deeplearning.analysis.LatentSpaceVisualizer;
import javax.swing.*;
import org.jfree.chart.*;

public class VAELatentSpacePanel extends JPanel {

    private ChartPanel chartPanel;
    private JComboBox<String> projectionMethodCombo;
    private JComboBox<String> colorSchemeCombo;
    private LatentSpaceVisualizer visualizer;

    public VAELatentSpacePanel() {
        setLayout(new BorderLayout());

        // Control panel
        JPanel controlPanel = createControlPanel();
        add(controlPanel, BorderLayout.NORTH);

        // Chart area
        chartPanel = new ChartPanel(null);
        add(chartPanel, BorderLayout.CENTER);
    }

    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new FlowLayout());

        panel.add(new JLabel("Projection:"));
        projectionMethodCombo = new JComboBox<>(new String[]{
            "t-SNE", "UMAP", "PCA", "Raw (if 2D)"
        });
        projectionMethodCombo.addActionListener(e -> updateVisualization());
        panel.add(projectionMethodCombo);

        panel.add(new JLabel("Color by:"));
        colorSchemeCombo = new JComboBox<>(new String[]{
            "Class", "Cluster", "Uncertainty", "Reconstruction Error"
        });
        colorSchemeCombo.addActionListener(e -> updateVisualization());
        panel.add(colorSchemeCombo);

        JButton exportButton = new JButton("Export");
        exportButton.addActionListener(e -> exportVisualization());
        panel.add(exportButton);

        return panel;
    }

    private void updateVisualization() {
        // Refresh chart with selected settings
        String projection = (String) projectionMethodCombo.getSelectedItem();
        String colorScheme = (String) colorSchemeCombo.getSelectedItem();

        JFreeChart chart = visualizer.create2DProjection(
            getLatentData(),
            parseProjectionMethod(projection),
            parseColorScheme(colorScheme)
        );

        chartPanel.setChart(chart);
    }

    private void exportVisualization() {
        // Export to file
    }
}
```

### 11.3 VAEReconstructionPanel.java

```java
package vtea.exploration.plottools.panels;

import ij.ImageStack;
import javax.swing.*;
import java.awt.*;

public class VAEReconstructionPanel extends JPanel {

    private JLabel originalLabel;
    private JLabel reconstructionLabel;
    private JLabel differenceLabel;
    private JTextArea metricsArea;

    public VAEReconstructionPanel() {
        setLayout(new GridLayout(1, 4));

        // Original image
        JPanel originalPanel = new JPanel(new BorderLayout());
        originalPanel.setBorder(BorderFactory.createTitledBorder("Original"));
        originalLabel = new JLabel();
        originalPanel.add(new JScrollPane(originalLabel), BorderLayout.CENTER);
        add(originalPanel);

        // Reconstruction
        JPanel reconPanel = new JPanel(new BorderLayout());
        reconPanel.setBorder(BorderFactory.createTitledBorder("Reconstruction"));
        reconstructionLabel = new JLabel();
        reconPanel.add(new JScrollPane(reconstructionLabel), BorderLayout.CENTER);
        add(reconPanel);

        // Difference
        JPanel diffPanel = new JPanel(new BorderLayout());
        diffPanel.setBorder(BorderFactory.createTitledBorder("Difference"));
        differenceLabel = new JLabel();
        diffPanel.add(new JScrollPane(differenceLabel), BorderLayout.CENTER);
        add(diffPanel);

        // Metrics
        JPanel metricsPanel = new JPanel(new BorderLayout());
        metricsPanel.setBorder(BorderFactory.createTitledBorder("Quality Metrics"));
        metricsArea = new JTextArea(10, 20);
        metricsArea.setEditable(false);
        metricsPanel.add(new JScrollPane(metricsArea), BorderLayout.CENTER);
        add(metricsPanel);
    }

    public void displayReconstruction(ImageStack original,
                                     ImageStack reconstruction,
                                     double mse, double ssim, double psnr) {
        // Display images
        originalLabel.setIcon(createIcon(original));
        reconstructionLabel.setIcon(createIcon(reconstruction));
        differenceLabel.setIcon(createDifferenceIcon(original, reconstruction));

        // Display metrics
        metricsArea.setText(String.format(
            "MSE: %.6f\nSSIM: %.4f\nPSNR: %.2f dB\n",
            mse, ssim, psnr
        ));
    }

    private ImageIcon createIcon(ImageStack stack) {
        // Convert ImageStack to displayable icon (max projection or slice)
        return null; // Placeholder
    }

    private ImageIcon createDifferenceIcon(ImageStack orig, ImageStack recon) {
        // Create difference image
        return null; // Placeholder
    }
}
```

---

<a name="implementation-sequence"></a>
## 12. Implementation Sequence

### Timeline Overview (Assuming 3D Classification Infrastructure Exists)

| **Sprint** | **Duration** | **Phase** | **Deliverables** |
|------------|-------------|-----------|------------------|
| 1 | Week 1-2 | Phase 1 | VAE core architecture (encoder, decoder, main model) |
| 2 | Week 2-3 | Phase 2 | Loss functions and training infrastructure |
| 3 | Week 4-5 | Phase 3 | Latent space extraction and visualization |
| 4 | Week 6 | Phase 4 | Reconstruction and quality metrics |
| 5 | Week 7 | Phase 5 | FeatureProcessing plugin integration |
| 6 | Week 8 | Phase 6 | Anomaly detection |
| 7 | Week 9-10 | Phase 7 | UI components |
| 8 | Week 10 | Testing | Comprehensive unit and integration tests |

### Detailed Sprint Breakdown

#### **Sprint 1: VAE Core Architecture (Week 1-2)**

**Tasks:**
1. Implement `VAEEncoder3D.java`
   - Convolutional blocks
   - Latent parameter outputs (μ, log σ²)
   - Weight initialization

2. Implement `VAEDecoder3D.java`
   - Transpose convolution blocks
   - Mirror encoder architecture
   - Sigmoid output activation

3. Implement `VariationalAutoencoder3D.java`
   - Combine encoder and decoder
   - Reparameterization trick
   - Forward/backward pass

4. Implement `BetaVAE3D.java`
   - Extend base VAE
   - β parameter handling

5. Create `VAEConfig.java`
   - Architecture configurations
   - Training hyperparameters
   - Serialization

**Testing:**
- Unit tests for encoder/decoder forward pass
- Test reparameterization trick
- Validate output dimensions

**Deliverable:** Working VAE model with forward pass

---

#### **Sprint 2: Loss Functions & Training (Week 2-3)**

**Tasks:**
1. Implement `ReconstructionLoss.java`
   - MSE, BCE, L1 variants

2. Implement `KLDivergenceLoss.java`
   - Analytical KL for Gaussians
   - Per-sample KL computation

3. Implement `VAELoss.java`
   - Combined loss
   - β-weighting
   - KL warmup schedule

4. Implement `VAEMetrics.java`
   - Track reconstruction, KL, ELBO
   - Early stopping logic

5. Implement `VAETrainer.java`
   - Training loop
   - Validation
   - Checkpointing

**Testing:**
- Test loss computation
- Validate gradient flow
- Test early stopping

**Deliverable:** Complete training pipeline

---

#### **Sprint 3: Latent Space Analysis (Week 4-5)**

**Tasks:**
1. Implement `VAELatentExtractor.java`
   - Extract latent representations
   - Uncertainty estimation

2. Implement `LatentSpaceVisualizer.java`
   - 2D/3D projections
   - t-SNE/UMAP integration
   - Color schemes

3. Implement `LatentSpaceInterpolator.java`
   - Linear interpolation
   - Spherical interpolation

**Testing:**
- Test latent extraction
- Validate interpolation
- Visual inspection of projections

**Deliverable:** Latent space analysis tools

---

#### **Sprint 4: Reconstruction & Quality (Week 6)**

**Tasks:**
1. Implement `VAEReconstructor.java`
   - Reconstruction pipeline
   - Quality metrics (MSE, SSIM, PSNR)

2. Implement `ReconstructionQualityAnalyzer.java`
   - Quality distributions
   - Outlier detection

**Testing:**
- Test reconstruction accuracy
- Validate quality metrics
- Test outlier detection

**Deliverable:** Reconstruction quality analysis

---

#### **Sprint 5: VTEA Integration (Week 7)**

**Tasks:**
1. Implement `VAEFeatureExtraction.java`
   - FeatureProcessing plugin
   - Latent features extraction

2. Implement `VAEClustering.java`
   - Clustering in latent space

3. Integration with existing VTEA workflows
   - MicroObject feature storage
   - Database persistence

**Testing:**
- Test plugin discovery
- Validate feature extraction
- Integration tests with VTEA

**Deliverable:** VTEA-compatible plugins

---

#### **Sprint 6: Anomaly Detection (Week 8)**

**Tasks:**
1. Implement `VAEAnomalyDetector.java`
   - Anomaly score computation
   - Threshold determination

2. Implement `VAEAnomalyDetection.java` (plugin)
   - Quality control workflow
   - Cell filtering

**Testing:**
- Test anomaly detection
- Validate threshold methods
- False positive/negative analysis

**Deliverable:** Anomaly detection system

---

#### **Sprint 7: UI Components (Week 9-10)**

**Tasks:**
1. Implement `VAETrainingPanel.java`
   - Training configuration UI
   - Progress monitoring

2. Implement `VAELatentSpacePanel.java`
   - Interactive latent space visualization

3. Implement `VAEReconstructionPanel.java`
   - Side-by-side reconstruction view
   - Quality metrics display

**Testing:**
- UI usability testing
- Integration with VTEA GUI

**Deliverable:** Complete UI for VAE functionality

---

#### **Sprint 8: Testing & Documentation (Week 10)**

**Tasks:**
1. Comprehensive unit tests
2. Integration tests
3. Performance benchmarking
4. User documentation
5. Code review and refactoring

**Deliverable:** Production-ready VAE system

---

<a name="use-cases"></a>
## 13. Use Cases in VTEA

### Use Case 1: Unsupervised Cell Type Discovery

**Scenario:** Researcher has 3D kidney tissue with unknown cell populations

**Workflow:**
1. Segment cells using existing VTEA tools
2. Train VAE on all segmented cells (unsupervised)
3. Extract latent representations
4. Visualize in 2D t-SNE projection
5. Identify clusters in latent space
6. Run K-means on latent space
7. Manually inspect representative cells from each cluster
8. Assign biological labels to clusters

**Benefits:**
- No manual labels required
- Learns from image data directly, not hand-crafted features
- Discovers structure automatically

---

### Use Case 2: Quality Control for Segmentation

**Scenario:** Researcher wants to filter out poorly segmented cells

**Workflow:**
1. Train VAE on high-quality manually curated cells
2. Run VAE reconstruction on all cells
3. Compute reconstruction error for each cell
4. Set threshold (e.g., 95th percentile)
5. Flag/remove cells with high reconstruction error
6. Proceed with downstream analysis on filtered dataset

**Benefits:**
- Automated quality control
- Objective filtering criteria
- Removes debris, artifacts, over-segmented cells

---

### Use Case 3: Transfer Learning for Classification

**Scenario:** Researcher has small labeled dataset, wants to improve classifier

**Workflow:**
1. Train VAE on large unlabeled dataset (all tissue types)
2. Extract latent features for labeled subset
3. Train shallow classifier (SVM, Random Forest) on latent features
4. Use for inference on new cells

**Benefits:**
- Leverage unlabeled data
- Better features than hand-crafted
- Requires fewer labeled examples

---

### Use Case 4: Rare Cell Detection

**Scenario:** Researcher looking for rare disease-associated cells

**Workflow:**
1. Train VAE on healthy tissue (normal cells)
2. Apply to diseased tissue
3. Compute anomaly scores
4. Cells with high scores are outliers
5. Manually inspect high-scoring cells
6. Identify novel cell states or rare populations

**Benefits:**
- Statistical rigor for outlier detection
- Adapts to normal variability
- Finds truly unusual cells

---

### Use Case 5: Data Augmentation

**Scenario:** Researcher needs more training examples for deep learning classifier

**Workflow:**
1. Train VAE on existing labeled cells
2. For minority classes, sample similar cells from latent space
3. Generate synthetic cells by decoding latent samples
4. Add to training set
5. Train classifier with augmented data

**Benefits:**
- Balances class distributions
- Creates realistic synthetic data
- Improves classifier generalization

---

### Use Case 6: Morphological Interpolation

**Scenario:** Researcher wants to understand transition between cell types

**Workflow:**
1. Select cell from type A and cell from type B
2. Interpolate in latent space
3. Decode intermediate latent vectors
4. Visualize morphological transition

**Benefits:**
- Interpretable latent space
- Understand gradual changes
- Hypothesis generation

---

<a name="testing"></a>
## 14. Testing Strategy

### 14.1 Unit Tests

**Model Tests:**
```java
// VAEEncoder3DTest.java
@Test
void testEncoderForwardPass() {
    VAEEncoder3D encoder = new VAEEncoder3D(1, 32, new int[]{32, 64, 128});
    Tensor input = rand(new long[]{4, 1, 64, 64, 64}); // Batch of 4

    VAEEncoder3D.EncoderOutput output = encoder.forward(input);

    assertEquals(4, output.mu.size(0));
    assertEquals(32, output.mu.size(1));
    assertEquals(4, output.logVar.size(0));
    assertEquals(32, output.logVar.size(1));
}

@Test
void testReparameterizationTrick() {
    VariationalAutoencoder3D vae = new VariationalAutoencoder3D(config);
    Tensor mu = zeros(new long[]{4, 32});
    Tensor logVar = ones(new long[]{4, 32}).mul(-2); // log(0.135)

    // Multiple samples should have variance
    List<Tensor> samples = new ArrayList<>();
    for (int i = 0; i < 100; i++) {
        samples.add(vae.reparameterize(mu, logVar));
    }

    // Check sample variance matches expected
    // ...
}
```

**Loss Tests:**
```java
// VAELossTest.java
@Test
void testKLDivergence() {
    KLDivergenceLoss klLoss = new KLDivergenceLoss();

    // KL(N(0,1) || N(0,1)) should be 0
    Tensor mu = zeros(new long[]{4, 32});
    Tensor logVar = zeros(new long[]{4, 32});
    Tensor kl = klLoss.compute(mu, logVar);

    assertTrue(kl.item().getDouble() < 1e-6);
}

@Test
void testReconstructionLoss() {
    ReconstructionLoss loss = new ReconstructionLoss(ReconstructionType.MSE);

    Tensor x = rand(new long[]{4, 1, 32, 32, 32});
    Tensor xRecon = x.clone();

    Tensor mse = loss.compute(xRecon, x);

    assertTrue(mse.item().getDouble() < 1e-6); // Should be near 0
}
```

### 14.2 Integration Tests

**End-to-End Training:**
```java
@Test
void testVAETraining() {
    // Create small synthetic dataset
    Tensor[] data = createSyntheticData(100, 1, 32, 32, 32);
    DataLoader loader = new DataLoader(data, 16);

    // Create VAE
    VAEConfig config = new VAEConfig();
    config.setLatentDim(16);
    config.setEpochs(5);
    VariationalAutoencoder3D vae = new VariationalAutoencoder3D(config);

    // Train
    VAETrainer trainer = new VAETrainer(vae, config, "./test_checkpoints");
    TrainingResult result = trainer.train(loader, loader);

    // Validate loss decreased
    assertTrue(result.getFinalLoss() < result.getInitialLoss());
}
```

**VTEA Plugin Integration:**
```java
@Test
void testVAEFeatureExtractionPlugin() {
    // Create mock MicroObjects
    ArrayList<MicroObject> objects = createMockObjects(50);
    ImageStack[] stacks = createMockImageStacks();

    // Run plugin
    VAEFeatureExtraction plugin = new VAEFeatureExtraction();
    plugin.processObjects(objects, stacks);

    // Validate features added
    for (MicroObject obj : objects) {
        assertFalse(obj.getFeatures().isEmpty());
        assertEquals(32, obj.getFeatures().get(0).size()); // Latent dim
    }
}
```

### 14.3 Performance Tests

**Inference Speed:**
```java
@Test
void testInferenceSpeed() {
    VariationalAutoencoder3D vae = loadPretrainedModel();
    Tensor input = rand(new long[]{1, 1, 64, 64, 64});

    long start = System.currentTimeMillis();
    for (int i = 0; i < 100; i++) {
        vae.forward(input);
    }
    long end = System.currentTimeMillis();

    double avgTime = (end - start) / 100.0;
    assertTrue(avgTime < 500); // Should be < 500ms per cell
}
```

**Memory Usage:**
```java
@Test
void testMemoryFootprint() {
    Runtime runtime = Runtime.getRuntime();
    long before = runtime.totalMemory() - runtime.freeMemory();

    VariationalAutoencoder3D vae = new VariationalAutoencoder3D(config);
    Tensor input = rand(new long[]{32, 1, 64, 64, 64}); // Large batch
    vae.forward(input);

    long after = runtime.totalMemory() - runtime.freeMemory();
    long used = (after - before) / 1024 / 1024; // MB

    assertTrue(used < 2048); // Should use < 2GB
}
```

### 14.4 Validation Tests

**Reconstruction Quality:**
```java
@Test
void testReconstructionQuality() {
    // Train on simple shapes (spheres, cubes)
    // Validate reconstructions are accurate

    VariationalAutoencoder3D vae = trainOnShapes();
    Tensor sphere = createSphere(64);

    VAEOutput output = vae.forward(sphere);
    double mse = computeMSE(sphere, output.reconstruction);

    assertTrue(mse < 0.01); // Good reconstruction
}
```

**Latent Space Structure:**
```java
@Test
void testLatentSpaceSmoothing() {
    // Interpolations should be smooth
    VariationalAutoencoder3D vae = loadPretrainedModel();

    Tensor cell1 = loadCell("cell1.tif");
    Tensor cell2 = loadCell("cell2.tif");

    Tensor[] interpolations = vae.interpolate(cell1, cell2, 10);

    // Check smoothness (adjacent interpolations should be similar)
    for (int i = 0; i < interpolations.length - 1; i++) {
        double diff = computeMSE(interpolations[i], interpolations[i + 1]);
        assertTrue(diff < 0.1); // Smooth transition
    }
}
```

---

<a name="challenges"></a>
## 15. Potential Challenges & Solutions

### Challenge 1: Posterior Collapse

**Problem:** KL divergence goes to 0, latent code ignored, model becomes deterministic autoencoder

**Solutions:**
- KL warmup: Gradually increase KL weight from 0 to β
- Free bits: Allow minimum KL per latent dimension
- β-VAE: Use β < 1 initially, increase later
- Monitor KL per dimension during training

**Implementation:**
```java
// In VAELoss.java
private double getKLWeight() {
    if (currentEpoch < warmupEpochs) {
        return beta * (double) currentEpoch / warmupEpochs;
    }
    return beta;
}

// Free bits
private Tensor applyFreeBits(Tensor kl, double freeBits) {
    return kl.clamp_min(freeBits);
}
```

---

### Challenge 2: Mode Collapse in Latent Space

**Problem:** All cells mapped to same region of latent space

**Solutions:**
- Increase latent dimensionality
- Use stronger regularization (higher β)
- Add auxiliary losses (e.g., MMD, adversarial)
- Check encoder capacity (may be too small)

**Monitoring:**
```java
// Compute latent space variance
double[] latentStds = computeLatentStandardDeviations(latents);
for (double std : latentStds) {
    if (std < 0.01) {
        logger.warn("Low variance in latent dimension: {}", std);
    }
}
```

---

### Challenge 3: Blurry Reconstructions

**Problem:** MSE loss leads to blurry outputs

**Solutions:**
- Use perceptual loss instead of MSE
- Add adversarial loss (VAE-GAN)
- Use L1 loss instead of MSE
- Add gradient penalty

**Implementation:**
```java
// Switch to L1 loss
config.setReconstructionType(ReconstructionType.L1);

// Or implement perceptual loss
public class PerceptualLoss {
    private NephNet3D featureExtractor; // Pre-trained

    public Tensor compute(Tensor recon, Tensor target) {
        Tensor reconFeatures = featureExtractor.extractFeatures(recon);
        Tensor targetFeatures = featureExtractor.extractFeatures(target);
        return mse_loss(reconFeatures, targetFeatures);
    }
}
```

---

### Challenge 4: Training Instability

**Problem:** Loss fluctuates wildly, NaN values

**Solutions:**
- Gradient clipping
- Reduce learning rate
- Batch normalization
- Monitor gradient norms
- Check data normalization

**Implementation:**
```java
// In VAETrainer.java
private void clipGradients(double maxNorm) {
    nn.utils.clip_grad_norm_(model.parameters(), maxNorm);
}

// Monitor gradients
private double getGradientNorm() {
    double totalNorm = 0;
    for (Parameter param : model.parameters()) {
        if (param.grad() != null) {
            totalNorm += param.grad().norm(2.0).pow(2.0).item().getDouble();
        }
    }
    return Math.sqrt(totalNorm);
}
```

---

### Challenge 5: Memory Constraints for Large Volumes

**Problem:** 128³ volumes with large batches exceed GPU memory

**Solutions:**
- Reduce batch size
- Use gradient accumulation
- Mixed precision training (FP16)
- Crop volumes to smaller regions
- Use checkpointing (trade compute for memory)

**Implementation:**
```java
// Gradient accumulation
int accumulationSteps = 4;
for (int i = 0; i < accumulationSteps; i++) {
    Batch batch = loader.next();
    VAEOutput output = model.forward(batch.getData());
    Tensor loss = lossFunction.compute(...).totalLoss;
    loss.div(accumulationSteps).backward();
}
optimizer.step();
optimizer.zero_grad();
```

---

### Challenge 6: Overfitting on Small Datasets

**Problem:** Limited training data (< 1000 cells)

**Solutions:**
- Strong data augmentation
- Reduce model capacity
- Early stopping
- Dropout
- Weight decay

**Implementation:**
```java
// In DataLoader.java
public Tensor augment(Tensor volume) {
    // Random 3D rotation
    volume = rotate3D(volume, randomAngle());

    // Random flip
    if (Math.random() > 0.5) volume = flip3D(volume, randomAxis());

    // Gaussian noise
    volume = volume.add(randn_like(volume).mul(0.01));

    // Brightness/contrast
    volume = volume.mul(1 + (Math.random() - 0.5) * 0.2);

    return volume;
}
```

---

### Challenge 7: Inconsistent Cell Sizes

**Problem:** Cells vary from 20³ to 150³ voxels

**Solutions:**
- Resize all to fixed size (with interpolation)
- Crop to bounding box + padding
- Multi-scale architecture
- Separate models for size ranges

**Implementation:**
```java
// In CellRegionExtractor.java
public Tensor extractRegion(MicroObject cell, ImageStack[] stacks,
                           int targetSize) {
    // Get bounding box
    BoundingBox bbox = cell.getBoundingBox();

    // Extract with padding
    Tensor region = extractWithPadding(stacks, bbox, targetSize);

    // Resize to target size
    if (region.size(2) != targetSize) {
        region = interpolate3D(region, targetSize, InterpolationMode.TRILINEAR);
    }

    return region;
}
```

---

### Challenge 8: Interpretability of Latent Dimensions

**Problem:** Hard to understand what each latent dimension encodes

**Solutions:**
- Use β-VAE for disentanglement
- Compute correlation with known features
- Latent traversal visualization
- Train linear probes

**Implementation:**
```java
public class LatentInterpretability {

    /**
     * Traverse single latent dimension
     */
    public List<ImageStack> traverseDimension(int dim, double min, double max,
                                             int steps) {
        List<ImageStack> traversal = new ArrayList<>();
        Tensor z = zeros(new long[]{1, latentDim});

        for (int i = 0; i < steps; i++) {
            double value = min + (max - min) * i / (steps - 1);
            z.select(1, dim).fill_(value);

            Tensor reconstruction = model.decode(z);
            traversal.add(tensorToImageStack(reconstruction));
        }

        return traversal;
    }

    /**
     * Compute correlation with morphological features
     */
    public double[] correlateWithFeatures(
            Map<MicroObject, float[]> latents,
            String featureName) {
        // Extract feature values
        // Compute Pearson correlation for each latent dim
        // Return correlation coefficients
    }
}
```

---

### Challenge 9: Batch Effects Between Datasets

**Problem:** VAE trained on one dataset doesn't generalize to another

**Solutions:**
- Train on combined datasets
- Domain adaptation techniques
- Conditional VAE with dataset ID
- Adversarial domain alignment

**Implementation:**
```java
public class DomainAdaptiveVAE extends VariationalAutoencoder3D {
    private Linear domainClassifier;
    private double adversarialWeight = 0.1;

    @Override
    public VAEOutput forward(Tensor x, Tensor domainLabel) {
        // Standard VAE forward
        VAEOutput output = super.forward(x);

        // Domain classifier on latent (with gradient reversal)
        Tensor domainPrediction = domainClassifier.forward(
            gradientReversal(output.z)
        );

        // Add adversarial loss to make latent domain-invariant
        // ...

        return output;
    }
}
```

---

## 16. Success Metrics

**Quantitative:**
- Reconstruction error (MSE, SSIM) on validation set
- KL divergence within expected range (2-10 per dimension)
- Latent space coverage (variance > 0.1 per dimension)
- Clustering metrics in latent space (silhouette score > 0.3)
- Anomaly detection precision/recall (F1 > 0.8)
- Inference speed (< 500ms per cell on GPU)

**Qualitative:**
- Visual quality of reconstructions
- Smoothness of latent space interpolations
- Separation of known cell types in latent space
- User feedback from VTEA integration

---

## 17. Next Steps After Implementation

1. **User Testing**
   - Recruit beta testers from research groups
   - Collect feedback on usability
   - Iterate on UI design

2. **Model Zoo**
   - Provide pre-trained models for common tissues
   - Enable transfer learning

3. **Documentation**
   - User guide with tutorials
   - API documentation
   - Example workflows

4. **Publication**
   - Write methods paper
   - Benchmark against existing methods
   - Share on BioRxiv

5. **Advanced Features**
   - Hierarchical VAE for multi-scale analysis
   - Video VAE for time-series
   - Graph VAE for spatial relationships

---

## Appendix A: Configuration Examples

**Small VAE (Fast):**
```json
{
  "architecture": "SMALL",
  "latentDim": 16,
  "encoderChannels": [16, 32, 64, 128],
  "inputSize": 32,
  "numChannels": 1,
  "learningRate": 1e-4,
  "batchSize": 32,
  "epochs": 100,
  "beta": 1.0,
  "reconstructionType": "MSE",
  "useKLWarmup": true,
  "warmupEpochs": 10
}
```

**Large β-VAE (Disentangled):**
```json
{
  "architecture": "LARGE",
  "latentDim": 64,
  "encoderChannels": [32, 64, 128, 256, 512],
  "inputSize": 128,
  "numChannels": 3,
  "learningRate": 5e-5,
  "batchSize": 8,
  "epochs": 200,
  "beta": 4.0,
  "reconstructionType": "L1",
  "useKLWarmup": true,
  "warmupEpochs": 20
}
```

---

## Appendix B: Comparison to 3D Classification

| **Aspect** | **3D Classification** | **3D VAE** |
|------------|----------------------|-----------|
| **Learning Type** | Supervised | Unsupervised |
| **Requires Labels** | Yes | No |
| **Output** | Class probabilities | Reconstructions + latent code |
| **Loss Function** | Cross-entropy | Reconstruction + KL |
| **Use Case** | Assign known cell types | Discover new types, QC |
| **Latent Space** | No explicit latent space | Continuous latent manifold |
| **Generation** | Not possible | Can generate new cells |
| **Uncertainty** | Via dropout/ensemble | Built-in (variance) |

**When to use each:**
- **Classification**: You have labeled data and want to assign cells to known types
- **VAE**: You want to explore unlabeled data, discover structure, or quality control

---

**END OF IMPLEMENTATION PLAN**
