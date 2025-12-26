# VAE Training Guide for VTEA

**Version:** 1.0
**Date:** 2025-12-26
**Branch:** `claude/add-3d-vae-vtea-KHwVA`

---

## Overview

This guide explains how to train a 3D Variational Autoencoder (VAE) on VTEA cell data using the implemented training infrastructure.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Pipeline](#training-pipeline)
3. [Configuration](#configuration)
4. [Data Preparation](#data-preparation)
5. [Running Training](#running-training)
6. [Monitoring Progress](#monitoring-progress)
7. [Model Evaluation](#model-evaluation)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Minimal Training Example

```java
import vtea.deeplearning.models.*;
import vtea.deeplearning.training.*;
import vtea.deeplearning.data.*;
import vteaobjects.MicroObject;
import ij.ImageStack;
import java.util.List;

// 1. Prepare your data
List<MicroObject> allCells = ...; // Your segmented cells
ImageStack[] imageStacks = ...; // Your multi-channel images

// Split into train/val (80/20)
int trainSize = (int) (allCells.size() * 0.8);
List<MicroObject> trainCells = allCells.subList(0, trainSize);
List<MicroObject> valCells = allCells.subList(trainSize, allCells.size());

// 2. Create configuration
VAEConfig config = new VAEConfig(VAEConfig.VAEArchitecture.MEDIUM);
config.setEpochs(100);
config.setBatchSize(16);
config.setLearningRate(1e-4);
config.setBeta(1.0);
config.setUseKLWarmup(true);

// 3. Create model
VariationalAutoencoder3D vae = new VariationalAutoencoder3D(config);

// 4. Create data loaders
DataLoader trainLoader = new DataLoader(
    trainCells, imageStacks,
    config.getBatchSize(), true, true, 64,
    TensorConverter.NormalizationType.ZSCORE, 42L
);

DataLoader valLoader = new DataLoader(
    valCells, imageStacks,
    config.getBatchSize(), false, false, 64,
    TensorConverter.NormalizationType.ZSCORE, 42L
);

// 5. Create trainer
VAETrainer trainer = new VAETrainer(vae, config, "./checkpoints");

// 6. Train!
VAETrainer.TrainingResult result = trainer.train(trainLoader, valLoader);

// 7. Check results
System.out.println(result.getMetrics().getSummary());
```

---

## Training Pipeline

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│ Data Preparation                                     │
│  - Load MicroObjects (cells)                        │
│  - Load ImageStacks (multi-channel images)          │
│  - Split train/validation                           │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ DataLoader                                          │
│  - Extract 3D regions (64³)                         │
│  - Convert to PyTorch tensors                       │
│  - Apply normalization (Z-score)                    │
│  - Batch creation (batch_size=16)                   │
│  - Data augmentation (rotation, flip, noise)        │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Training Loop (VAETrainer)                          │
│                                                     │
│  For each epoch:                                    │
│    1. Training phase:                               │
│       - Forward pass through VAE                    │
│       - Compute loss (Recon + β×KL)                │
│       - Backward pass                               │
│       - Optimizer step (Adam)                       │
│                                                     │
│    2. Validation phase:                             │
│       - Forward pass (no gradients)                 │
│       - Compute validation loss                     │
│                                                     │
│    3. Metrics & checkpointing:                      │
│       - Log epoch metrics                           │
│       - Save checkpoint if best                     │
│       - Check early stopping                        │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ Results                                             │
│  - Trained model                                    │
│  - Training metrics (loss curves)                   │
│  - Best checkpoint saved                            │
│  - Metrics CSV for plotting                         │
└─────────────────────────────────────────────────────┘
```

---

## Configuration

### VAE Architecture Choices

| Architecture | Input Size | Latent Dim | GPU Memory | Use Case |
|-------------|-----------|-----------|------------|----------|
| **SMALL** | 32³ | 16 | ~2 GB | Fast prototyping, limited data |
| **MEDIUM** | 64³ | 32 | ~4 GB | **Recommended** for most datasets |
| **LARGE** | 128³ | 64 | ~8 GB | High-resolution cells, lots of data |

### Configuration Options

```java
VAEConfig config = new VAEConfig();

// Architecture
config.setArchitecture(VAEConfig.VAEArchitecture.MEDIUM);
config.setLatentDim(32);                    // Latent space dimensions
config.setInputSize(64);                    // Input volume size (64³)
config.setNumChannels(1);                   // Grayscale = 1, RGB = 3

// Training hyperparameters
config.setLearningRate(1e-4);               // Adam learning rate
config.setBatchSize(16);                    // Batch size (reduce if OOM)
config.setEpochs(100);                      // Number of epochs

// Loss configuration
config.setReconstructionType(
    VAEConfig.ReconstructionType.MSE        // MSE, BCE, or L1
);
config.setBeta(1.0);                        // β-VAE parameter (1.0 = standard VAE)
config.setUseKLWarmup(true);                // Gradually increase KL weight
config.setWarmupEpochs(10);                 // Warmup over N epochs

// Data processing
config.setNormalization(
    VAEConfig.NormalizationType.ZSCORE      // Z-score, MINMAX, or NONE
);
config.setUseAugmentation(true);            // Enable data augmentation

// Device
config.setUseGPU(true);                     // Use GPU if available

// Save configuration
config.saveToFile("vae_config.json");
```

### Recommended Settings by Use Case

**Quick Prototyping:**
```java
config.setArchitecture(VAEArchitecture.SMALL);
config.setEpochs(20);
config.setBatchSize(32);
config.setUseAugmentation(false);
```

**Production Training (Classification):**
```java
config.setArchitecture(VAEArchitecture.MEDIUM);
config.setEpochs(150);
config.setBatchSize(16);
config.setLearningRate(5e-5);
config.setBeta(1.0);
config.setUseKLWarmup(true);
config.setUseAugmentation(true);
```

**Disentangled Representations (β-VAE):**
```java
config.setArchitecture(VAEArchitecture.MEDIUM);
config.setBeta(4.0);               // Higher β for more disentanglement
config.setUseKLWarmup(false);      // No warmup for β-VAE
```

---

## Data Preparation

### 1. Load Cells from VTEA

```java
// Assuming you have VTEA's H2DatabaseEngine
import vtea.jdbc.H2DatabaseEngine;

// Load cells from database
List<MicroObject> allCells = loadCellsFromDatabase();

// Filter cells (optional)
List<MicroObject> filteredCells = allCells.stream()
    .filter(cell -> cell.getVolume() > 100)  // Minimum size
    .filter(cell -> !cell.isExcluded())      // Not manually excluded
    .collect(Collectors.toList());

System.out.printf("Loaded %d cells\n", filteredCells.size());
```

### 2. Load Image Data

```java
import ij.ImagePlus;
import ij.ImageStack;

// Load multi-channel image
ImagePlus imp = IJ.openImage("path/to/image.tif");

// Get stacks for each channel
ImageStack[] imageStacks = new ImageStack[imp.getNChannels()];
for (int c = 0; c < imp.getNChannels(); c++) {
    imp.setC(c + 1);
    imageStacks[c] = imp.getImageStack().duplicate();
}
```

### 3. Train/Validation Split

```java
// Shuffle cells
Collections.shuffle(allCells, new Random(42));

// 80/20 split
int trainSize = (int) (allCells.size() * 0.8);
List<MicroObject> trainCells = allCells.subList(0, trainSize);
List<MicroObject> valCells = allCells.subList(trainSize, allCells.size());

System.out.printf("Train: %d cells, Val: %d cells\n",
    trainCells.size(), valCells.size());
```

### 4. Create DataLoaders

```java
DataLoader trainLoader = new DataLoader(
    trainCells,                              // Cell list
    imageStacks,                             // Image data
    config.getBatchSize(),                   // Batch size
    true,                                    // Shuffle
    config.isUseAugmentation(),              // Augmentation
    config.getInputSize(),                   // Region size (64)
    config.getNormalization(),               // Normalization type
    config.getRandomSeed()                   // Random seed
);

DataLoader valLoader = new DataLoader(
    valCells, imageStacks,
    config.getBatchSize(),
    false,    // Don't shuffle validation
    false,    // No augmentation for validation
    config.getInputSize(),
    config.getNormalization(),
    config.getRandomSeed()
);

System.out.printf("Train batches: %d, Val batches: %d\n",
    trainLoader.size(), valLoader.size());
```

---

## Running Training

### Basic Training

```java
// Create trainer
VAETrainer trainer = new VAETrainer(model, config, "./checkpoints");

// Run training
VAETrainer.TrainingResult result = trainer.train(trainLoader, valLoader);

// Print summary
System.out.println(result.getMetrics().getSummary());
```

### Training with Progress Updates

```java
import vtea.processor.listeners.ProgressListener;

// Create trainer
VAETrainer trainer = new VAETrainer(model, config, "./checkpoints");

// Add progress listener
trainer.addProgressListener(new ProgressListener() {
    @Override
    public void FireProgressChange(String message, double progress) {
        System.out.printf("[%3.0f%%] %s\n", progress * 100, message);
    }
});

// Train
VAETrainer.TrainingResult result = trainer.train(trainLoader, valLoader);
```

### Training with UI Integration (VTEA Panel)

```java
// In a VTEA UI component
public class VAETrainingPanel extends JPanel {

    private JProgressBar progressBar;
    private JTextArea logArea;
    private VAETrainer trainer;

    public void startTraining() {
        // Create trainer
        trainer = new VAETrainer(model, config, "./checkpoints");

        // Add progress listener
        trainer.addProgressListener((message, progress) -> {
            SwingUtilities.invokeLater(() -> {
                progressBar.setValue((int) (progress * 100));
                logArea.append(message + "\n");
            });
        });

        // Run in background thread
        new SwingWorker<VAETrainer.TrainingResult, Void>() {
            @Override
            protected VAETrainer.TrainingResult doInBackground() {
                return trainer.train(trainLoader, valLoader);
            }

            @Override
            protected void done() {
                try {
                    VAETrainer.TrainingResult result = get();
                    JOptionPane.showMessageDialog(null,
                        result.getMetrics().getSummary(),
                        "Training Complete",
                        JOptionPane.INFORMATION_MESSAGE);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }.execute();
    }

    public void stopTraining() {
        if (trainer != null) {
            trainer.stop();
        }
    }
}
```

---

## Monitoring Progress

### Console Output

Training produces detailed console output:

```
[INFO] VAETrainer - Starting VAE training: 100 epochs, 125 batches/epoch
[INFO] VAETrainer - Epoch 0: Train Loss=0.4521 (Recon=0.3812, KL=0.0709, ELBO=-0.4521), Val Loss=0.4201 (Recon=0.3598, KL=0.0603, ELBO=-0.4201)
[INFO] ModelCheckpoint - New best model saved (val loss: 0.4201)
[INFO] VAETrainer - Epoch 1: Train Loss=0.3987 (Recon=0.3401, KL=0.0586, ELBO=-0.3987), Val Loss=0.3856 (Recon=0.3312, KL=0.0544, ELBO=-0.3856)
...
```

### Metrics Tracking

```java
// Access metrics during/after training
TrainingMetrics metrics = trainer.getMetrics();

// Get training history
List<TrainingMetrics.EpochMetrics> trainHistory = metrics.getTrainHistory();
List<TrainingMetrics.EpochMetrics> valHistory = metrics.getValHistory();

// Plot loss curves (example)
for (int i = 0; i < trainHistory.size(); i++) {
    TrainingMetrics.EpochMetrics train = trainHistory.get(i);
    TrainingMetrics.EpochMetrics val = valHistory.get(i);

    System.out.printf("Epoch %d: Train=%.4f, Val=%.4f\n",
        i, train.totalLoss, val.totalLoss);
}

// Export to CSV for plotting
metrics.saveToCSV("training_metrics.csv");
```

### Checkpoints

Checkpoints are saved automatically to `./checkpoints/checkpoint_epochXXX_timestamp/`:

```
checkpoints/
├── checkpoint_epoch010_20251226_143052/
│   ├── model.pt           # PyTorch model weights
│   ├── config.json        # VAE configuration
│   ├── metadata.json      # Training metadata
│   └── metrics.csv        # Training history
├── checkpoint_epoch025_20251226_144312/
└── checkpoint_epoch042_20251226_145821/  # Best model
```

### Loading Checkpoints

```java
// Load latest checkpoint
ModelCheckpoint checkpoint = new ModelCheckpoint("./checkpoints");
String latestPath = checkpoint.findLatestCheckpoint();

if (latestPath != null) {
    // Load model
    VariationalAutoencoder3D loadedModel = checkpoint.load(latestPath);

    // Load metadata
    Map<String, Object> metadata = checkpoint.loadMetadata(latestPath);
    System.out.println("Loaded model from epoch: " + metadata.get("epoch"));
    System.out.println("Val loss: " + metadata.get("val_loss"));
}
```

---

## Model Evaluation

### Extract Latent Features

```java
// Set model to eval mode
model.eval();

// Process all cells
List<float[]> latentFeatures = new ArrayList<>();
for (MicroObject cell : allCells) {
    // Extract region
    ImageStack region = extractor.extractRegion(cell, imageStacks[0]);

    // Convert to tensor
    Tensor input = converter.imageStackToTensor(region);

    // Get latent representation
    VAEEncoder3D.EncoderOutput encoded = model.encode(input);

    // Convert to float array
    float[] latent = tensorToFloatArray(encoded.mu);
    latentFeatures.add(latent);
}

System.out.printf("Extracted latent features for %d cells\n",
    latentFeatures.size());
```

### Reconstruction Quality

```java
import vtea.deeplearning.inference.VAEReconstructor;

VAEReconstructor reconstructor = new VAEReconstructor(model, extractor);

// Reconstruct all cells
Map<MicroObject, VAEReconstructor.ReconstructionResult> results =
    reconstructor.reconstructAll(cells, imageStacks);

// Find poor reconstructions
List<MicroObject> lowQuality =
    reconstructor.identifyLowQualityCells(results, 0.05); // MSE threshold

System.out.printf("Found %d low-quality cells\n", lowQuality.size());

// Inspect individual reconstruction
for (Map.Entry<MicroObject, VAEReconstructor.ReconstructionResult> entry :
     results.entrySet()) {

    ReconstructionResult result = entry.getValue();
    System.out.printf("Cell: MSE=%.6f, SSIM=%.4f, PSNR=%.2f\n",
        result.mse, result.ssim, result.psnr);
}
```

### Classification with Latent Features

```java
import smile.clustering.KMeans;

// K-Means clustering on latent features
double[][] data = convertToMatrix(latentFeatures);
KMeans kmeans = new KMeans(data, 5); // 5 clusters

// Get cluster assignments
int[] clusters = kmeans.getClusterLabel();

// Assign to cells
for (int i = 0; i < cells.size(); i++) {
    cells.get(i).setCluster(clusters[i]);
}
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** `OutOfMemoryError`, CUDA OOM

**Solutions:**
1. Reduce batch size:
   ```java
   config.setBatchSize(8); // Try 8, 4, or even 2
   ```

2. Use smaller architecture:
   ```java
   config.setArchitecture(VAEArchitecture.SMALL);
   ```

3. Reduce input size:
   ```java
   config.setInputSize(32); // Use 32³ instead of 64³
   ```

4. Use CPU instead of GPU:
   ```java
   config.setUseGPU(false);
   ```

### Loss is NaN or Infinite

**Symptoms:** `totalLoss=NaN` in logs

**Solutions:**
1. Reduce learning rate:
   ```java
   config.setLearningRate(1e-5); // Lower from 1e-4
   ```

2. Enable gradient clipping:
   ```java
   VAETrainer trainer = new VAETrainer(model, config, "./checkpoints",
       true,  // Enable gradient clipping
       1.0    // Max gradient norm
   );
   ```

3. Check data normalization:
   ```java
   // Ensure Z-score normalization is used
   config.setNormalization(NormalizationType.ZSCORE);
   ```

### Posterior Collapse (KL → 0)

**Symptoms:** KL divergence drops to near-zero

**Solutions:**
1. Use KL warmup:
   ```java
   config.setUseKLWarmup(true);
   config.setWarmupEpochs(20); // Longer warmup
   ```

2. Increase β temporarily:
   ```java
   config.setBeta(2.0); // Higher KL weight
   ```

3. Monitor per-dimension KL:
   ```java
   KLDivergenceLoss klLoss = new KLDivergenceLoss();
   Tensor klPerDim = klLoss.computePerDimension(mu, logVar);
   // Check which dimensions have collapsed
   ```

### Training Not Improving

**Symptoms:** Loss plateaus early

**Solutions:**
1. Reduce learning rate:
   ```java
   config.setLearningRate(5e-5);
   ```

2. Increase model capacity:
   ```java
   config.setArchitecture(VAEArchitecture.LARGE);
   config.setLatentDim(64);
   ```

3. More training data:
   - Collect more cells
   - Enable stronger augmentation

4. Check for data issues:
   - Verify cells are properly segmented
   - Check for duplicates or corrupted data

### Blurry Reconstructions

**Symptoms:** Reconstructions lack detail

**Solutions:**
1. Try L1 loss instead of MSE:
   ```java
   config.setReconstructionType(ReconstructionType.L1);
   ```

2. Reduce β (less regularization):
   ```java
   config.setBeta(0.5);
   ```

3. Increase latent dimensions:
   ```java
   config.setLatentDim(64); // More capacity
   ```

---

## Advanced Topics

### Learning Rate Scheduling

```java
// Implement custom trainer with LR decay
public class VAETrainerWithScheduler extends VAETrainer {

    private double initialLR;
    private double decayRate = 0.95;
    private int decayEvery = 10; // Epochs

    @Override
    protected void onEpochEnd(int epoch) {
        if (epoch > 0 && epoch % decayEvery == 0) {
            double newLR = initialLR * Math.pow(decayRate, epoch / decayEvery);
            optimizer.setLearningRate(newLR);
            logger.info("Learning rate decayed to: {}", newLR);
        }
    }
}
```

### Transfer Learning

```java
// Load pre-trained encoder, train new decoder
VariationalAutoencoder3D pretrained = checkpoint.load("./pretrained");

// Freeze encoder
pretrained.getEncoder().eval();
pretrained.getEncoder().requires_grad_(false);

// Train only decoder
VAETrainer trainer = new VAETrainer(pretrained, config, "./checkpoints");
trainer.train(trainLoader, valLoader);
```

### Conditional VAE (by Cell Type)

```java
// Add class conditioning (future implementation)
// This is a placeholder for future conditional VAE support

Map<MicroObject, Integer> cellClasses = ...; // Cell → class ID

// Train conditional VAE
ConditionalVAE3D cvae = new ConditionalVAE3D(config, numClasses);
// ... training with class labels
```

---

## Performance Tips

### Speed Optimization

1. **Use GPU:** 10-50x faster than CPU
   ```java
   config.setUseGPU(true);
   ```

2. **Larger batch sizes:** Better GPU utilization
   ```java
   config.setBatchSize(32); // If memory allows
   ```

3. **Disable augmentation during validation:**
   ```java
   valLoader = new DataLoader(..., false, false, ...); // No augmentation
   ```

4. **Pre-extract regions:** Cache extracted regions to disk (future)

### Memory Optimization

1. **Mixed precision training:** (Future - if JavaCPP supports it)

2. **Gradient accumulation:** Simulate larger batches
   ```java
   // Modify trainer to accumulate gradients over N batches
   ```

3. **Checkpoint CPU offloading:** Keep model on CPU between epochs

---

## Summary

This training infrastructure provides:

✅ **Complete training pipeline** - Data loading to model saving
✅ **VTEA integration** - Compatible with existing patterns
✅ **Progress tracking** - Real-time updates for UI
✅ **Automatic checkpointing** - Never lose progress
✅ **Early stopping** - Prevent overfitting
✅ **Data augmentation** - Improve generalization
✅ **Comprehensive logging** - Debug and monitor
✅ **Production-ready** - Robust error handling

**Next Steps:**
1. Train your first model with example code
2. Evaluate reconstruction quality
3. Use latent features for classification
4. Integrate with VTEA workflow

For questions or issues, check the [troubleshooting section](#troubleshooting) or review the implementation code.
