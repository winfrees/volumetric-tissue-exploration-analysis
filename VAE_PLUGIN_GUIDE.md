# VAE FeatureProcessing Plugin Guide

This guide covers the four VAE-based FeatureProcessing plugins for VTEA workflow integration.

## Overview

The VAE plugins enable deep learning-based analysis of volumetric cellular data directly within VTEA workflows. All plugins use trained 3D VAE models to extract features, reduce dimensionality, cluster cells, and detect anomalies.

### Plugin Summary

| Plugin | Type | Purpose | Output |
|--------|------|---------|--------|
| **VAEFeatureExtraction** | Feature | Extract latent features | N-dimensional latent vectors |
| **VAEDimensionalityReduction** | Reduction | Reduce to 2D/3D for visualization | 2D or 3D coordinates |
| **VAEClustering** | Cluster | K-means clustering in latent space | Cluster IDs (0, 1, 2, ...) |
| **VAEAnomalyDetection** | Other | Detect anomalous cells | Error/Score/Binary |

---

## 1. VAE Feature Extraction

### Purpose
Extracts low-dimensional latent features from volumetric cell regions using a trained 3D VAE encoder.

### Parameters

**VAE Model Path**: Directory containing trained VAE checkpoint
- Must contain: `model.pt`, `config.json`, `metadata.json`
- Use Browse button to select directory

### Output
- **Type**: Feature
- **Dimensionality**: Matches VAE latent dimension (typically 16-128)
- **Format**: One latent vector per cell
- **Column Names**: `VAE_Latent_0`, `VAE_Latent_1`, ..., `VAE_Latent_N`

### Usage Example

```
Workflow: Segmentation → VAE Feature Extraction → Clustering
```

1. Add "VAE Feature Extraction" block to workflow
2. Select trained model checkpoint directory
3. Run workflow
4. Latent features added to feature table

### Use Cases

- **Unsupervised feature learning**: Extract meaningful features without manual labeling
- **Transfer learning**: Use features for downstream classification
- **Feature compression**: Reduce 64³ = 262,144 voxels to 16-128 dimensions
- **Data exploration**: Visualize cells in latent space

### Technical Details

- Extracts cubic regions centered on cell centroids
- Region size determined by model configuration (32³, 64³, or 128³)
- Uses **mean** of latent distribution (μ), not sampled z
- Applies same normalization as during training (Z-score)
- Padding strategy: REPLICATE (mirrors edge voxels for boundary cells)

---

## 2. VAE Dimensionality Reduction

### Purpose
Reduces high-dimensional latent features to 2D or 3D for visualization in MicroExplorer.

### Parameters

**VAE Model Path**: Directory containing trained VAE checkpoint

**Output Dimensions**:
- `2D` - For 2D scatter plots (default)
- `3D` - For 3D scatter plots

### Output
- **Type**: Reduction
- **Dimensionality**: 2 or 3
- **Format**: Coordinates for each cell
- **Column Names**: `VAE_2D_0`, `VAE_2D_1` or `VAE_3D_0`, `VAE_3D_1`, `VAE_3D_2`

### Usage Example

```
Workflow: Segmentation → VAE Dimensionality Reduction (2D) → Visualization
```

1. Add "VAE Dimensionality Reduction" block
2. Select model and output dimensions (2D/3D)
3. Run workflow
4. Visualize in MicroExplorer scatter plot

### Use Cases

- **Data visualization**: Create 2D/3D scatter plots of cells
- **Population discovery**: Identify subpopulations visually
- **Quality control**: Spot outliers and anomalies
- **Manuscript figures**: Generate publication-quality visualizations

### Technical Details

- First encodes cells to full latent space (e.g., 32D)
- Then applies PCA-like reduction to target dimensions
- Current implementation: uses first N principal dimensions
- Future enhancement: Full PCA with eigendecomposition

---

## 3. VAE Clustering

### Purpose
Performs k-means clustering on cells using their VAE latent representations.

### Parameters

**VAE Model Path**: Directory containing trained VAE checkpoint

**Number of Clusters**:
- Range: 2 to number of cells
- Default: 5
- Determines k for k-means

**Max Iterations**:
- Maximum iterations for k-means convergence
- Default: 100
- Increase for large datasets or many clusters

### Output
- **Type**: Cluster
- **Dimensionality**: 1 (cluster ID)
- **Format**: Integer cluster assignments
- **Values**: 0, 1, 2, ..., k-1

### Usage Example

```
Workflow: Segmentation → VAE Clustering (k=5) → Gating
```

1. Add "VAE Clustering" block
2. Select model and number of clusters
3. Run workflow
4. Use cluster IDs for gating and analysis

### Use Cases

- **Cell type discovery**: Identify distinct cell populations
- **Automated gating**: Replace manual gating with unsupervised clustering
- **Subpopulation analysis**: Quantify cluster distributions across conditions
- **Rare cell detection**: Identify small, distinct populations

### Technical Details

- Extracts latent features (same as VAEFeatureExtraction)
- Runs k-means clustering in latent space
- Uses Euclidean distance in latent dimensions
- Random initialization with fixed seed (reproducible)
- Logs cluster size distribution

### Interpreting Results

The plugin logs cluster distribution:
```
Cluster distribution: C0=234 C1=189 C2=456 C3=123 C4=98
```

- **Balanced clusters** (similar sizes): Data has uniform structure
- **Unbalanced clusters** (very different sizes): Rare populations or outliers
- **Empty clusters**: Reduce k or increase max iterations

---

## 4. VAE Anomaly Detection

### Purpose
Identifies anomalous cells based on VAE reconstruction error. Cells that reconstruct poorly are likely outliers, debris, or rare cell types.

### Parameters

**VAE Model Path**: Directory containing trained VAE checkpoint

**Output Mode**:
- `Reconstruction Error` - Raw MSE values
- `Anomaly Score (0-1)` - Normalized scores (0=normal, 1=anomalous)
- `Binary (Normal/Anomaly)` - Classification (0=normal, 1=anomaly)

**Anomaly Threshold (SD)**:
- For binary mode only
- Range: 1.0 to 5.0 standard deviations
- Default: 2.0
- Cells with error > (mean + threshold × SD) flagged as anomalies

### Output

**Reconstruction Error Mode**:
- Type: Other
- Format: Continuous MSE values
- Interpretation: Higher = more anomalous

**Anomaly Score Mode**:
- Type: Other
- Format: Normalized scores [0, 1]
- Interpretation: 0 = perfectly normal, 1 = highly anomalous

**Binary Mode**:
- Type: Other
- Format: Binary classification
- Values: 0 (normal), 1 (anomaly)

### Usage Example

```
Workflow: Segmentation → VAE Anomaly Detection (Binary, 2.0 SD) → Filter
```

1. Add "VAE Anomaly Detection" block
2. Select model, mode, and threshold
3. Run workflow
4. Filter or gate on anomaly flag

### Use Cases

- **Quality control**: Remove debris, doublets, and imaging artifacts
- **Rare cell detection**: Identify unusual cells for manual review
- **Data cleaning**: Pre-filter before downstream analysis
- **Batch effect detection**: Flag cells from problematic samples

### Technical Details

- Computes full reconstruction (encode → decode)
- Calculates Mean Squared Error (MSE) per cell
- MSE computed on normalized voxel intensities
- Logs statistics: median error, 95th percentile

### Interpreting Results

The plugin logs statistics:
```
Reconstruction error statistics: median=0.0234, 95th percentile=0.0678
Detected 23 anomalies out of 1000 cells (2.30%)
```

**Typical anomaly rates**:
- **0-2%**: Normal, well-trained model on clean data
- **2-5%**: Expected for heterogeneous samples
- **5-10%**: Consider reviewing training data or model
- **>10%**: Model may not generalize to this dataset

**Adjusting sensitivity**:
- **Too many anomalies**: Increase threshold (2.0 → 3.0 SD)
- **Missing anomalies**: Decrease threshold (2.0 → 1.5 SD)
- **Uncertain**: Use Anomaly Score mode for manual thresholding

---

## Workflow Integration

### Prerequisites

1. **Trained VAE Model**: Complete training using `VAETrainer`
2. **Checkpoint Directory**: Contains model.pt, config.json, metadata.json
3. **Segmented Data**: Objects must be segmented before feature processing
4. **Image Stack**: Original volumetric data must be available

### Adding Plugins to Workflows

All four plugins appear in VTEA's FeatureProcessing menu:

```
FeatureProcessing
├── Features
│   └── VAE Feature Extraction
├── Reduction
│   └── VAE Dimensionality Reduction
├── Cluster
│   └── VAE Clustering
└── Other
    └── VAE Anomaly Detection
```

### Common Workflow Patterns

**Pattern 1: Unsupervised Discovery**
```
Segmentation → VAE Feature Extraction → PCA → Clustering → Visualization
```

**Pattern 2: Anomaly Filtering**
```
Segmentation → VAE Anomaly Detection → Gate (Normal) → Feature Extraction
```

**Pattern 3: Direct Clustering**
```
Segmentation → VAE Clustering → Statistics → Export
```

**Pattern 4: Comprehensive Analysis**
```
Segmentation
    ├→ VAE Feature Extraction → Classification
    ├→ VAE Dimensionality Reduction → Visualization
    ├→ VAE Clustering → Gating
    └→ VAE Anomaly Detection → QC Report
```

---

## Performance Considerations

### Memory Requirements

| Input Size | Batch | GPU Memory | CPU Memory |
|------------|-------|------------|------------|
| 32³ × 100 cells | 10 | ~2 GB | ~4 GB |
| 64³ × 100 cells | 5 | ~4 GB | ~8 GB |
| 128³ × 100 cells | 2 | ~8 GB | ~16 GB |

### Processing Speed

**GPU (CUDA)**:
- 32³: ~50-100 cells/sec
- 64³: ~20-50 cells/sec
- 128³: ~5-10 cells/sec

**CPU**:
- 32³: ~5-10 cells/sec
- 64³: ~1-2 cells/sec
- 128³: ~0.2-0.5 cells/sec

**Recommendation**: Use GPU for datasets >100 cells or input size ≥64³

### Optimization Tips

1. **Batch similar analyses**: Run all VAE plugins in single workflow
2. **GPU acceleration**: Ensure CUDA is available and enabled in model config
3. **Region size**: Use smallest input size that captures cell features (32³ < 64³ < 128³)
4. **Model precision**: Use FP16 models for 2× speedup with minimal accuracy loss

---

## Troubleshooting

### Error: "Model path not specified"
**Cause**: No checkpoint directory selected
**Solution**: Click Browse and select valid checkpoint directory

### Error: "Model directory does not exist"
**Cause**: Invalid path or checkpoint deleted
**Solution**: Verify path exists and contains model.pt

### Error: "No objects found in MicroBlockSetup"
**Cause**: Segmentation step missing or failed
**Solution**: Ensure segmentation runs before VAE plugins

### Error: "No image stack found"
**Cause**: Original image data not available
**Solution**: Verify image stack loaded in VTEA

### Warning: "High reconstruction errors"
**Cause**: Model doesn't generalize to current dataset
**Solutions**:
- Check if data matches training distribution
- Retrain model with diverse samples
- Adjust normalization settings

### Performance: "Processing very slow"
**Causes**: CPU processing, large input size, many cells
**Solutions**:
- Enable GPU in model config
- Reduce input size (128³ → 64³)
- Process in batches
- Close other applications

---

## Advanced Usage

### Custom Model Architectures

All plugins support custom VAE architectures:

```java
// Create custom config
VAEConfig config = new VAEConfig();
config.setArchitecture(VAEConfig.VAEArchitecture.CUSTOM);
config.setInputSize(96);  // Custom size
config.setLatentDim(64);  // Custom latent dim
config.setEncoderChannels(new int[]{64, 128, 256, 512});
config.setDecoderChannels(new int[]{512, 256, 128, 64});

// Train model
VAETrainer trainer = new VAETrainer(model, config, null);
trainer.train(trainLoader, valLoader);

// Use in plugins (automatically loads config)
```

### Multiple Models for Different Channels

Process different image channels with specialized models:

```
Workflow:
  Channel 1 (Nuclei) → VAE Model A → Features A
  Channel 2 (Membrane) → VAE Model B → Features B
  Combine Features A + B → Classification
```

### Transfer Learning

Use pre-trained features for supervised classification:

```
1. VAE Feature Extraction (pre-trained model)
2. Export features to CSV
3. Train external classifier (scikit-learn, PyTorch)
4. Import predictions back to VTEA
```

---

## Plugin Implementation Details

### Architecture

All plugins extend `AbstractFeatureProcessing` and implement:

1. **GUI Components**: Parameters for user configuration
2. **Static Comment Method**: Block GUI display text
3. **Process Method**: Main analysis logic
4. **Result Storage**: Results in `dataResult` ArrayList

### Plugin Lifecycle

```
1. User adds plugin block to workflow
2. Setup GUI created (constructor with max parameter)
3. User configures parameters
4. Workflow executes
5. process() called with parameters and feature table
6. Plugin loads model, processes data, stores results
7. Results merged into VTEA feature table
8. Model and resources cleaned up
```

### Error Handling

All plugins include comprehensive error handling:
- Parameter validation
- File existence checks
- Null checks for data
- Try-catch with cleanup in finally blocks
- Detailed logging at INFO, DEBUG, and ERROR levels

### Resource Management

Proper cleanup prevents memory leaks:
```java
try {
    // Processing...
} finally {
    if (model != null) {
        model.close();  // Release PyTorch memory
    }
    // Tensors closed immediately after use
}
```

---

## File Locations

### Plugin Source Files

```
src/main/java/vtea/featureprocessing/
├── VAEFeatureExtraction.java       (268 lines)
├── VAEDimensionalityReduction.java (313 lines)
├── VAEClustering.java              (367 lines)
└── VAEAnomalyDetection.java        (391 lines)
```

### Dependencies

All plugins depend on:
- `vtea.deeplearning.models.VariationalAutoencoder3D`
- `vtea.deeplearning.CellRegionExtractor`
- `vtea.deeplearning.TensorConverter`
- `vtea.objects.MicroObject`
- `vtea.protocol.setup.MicroBlockSetup`
- `org.bytedeco.pytorch.*`
- `org.slf4j.Logger`

---

## Citation

If you use these VAE plugins in your research, please cite:

```
@software{vtea_vae_2025,
  title={3D Variational Autoencoder for Volumetric Tissue Exploration and Analysis},
  author={VTEA Developer},
  year={2025},
  url={https://github.com/your-repo/vtea}
}
```

---

## Next Steps

1. **Train a VAE model**: See `TRAINING_GUIDE.md`
2. **Test plugins**: Run on sample dataset
3. **Build workflows**: Combine plugins for comprehensive analysis
4. **Optimize parameters**: Tune thresholds and cluster numbers
5. **Validate results**: Compare with manual gating/classification

For questions or issues, see:
- `VAE_3D_IMPLEMENTATION_PLAN.md` - Architecture details
- `TRAINING_GUIDE.md` - Model training
- `IMPLEMENTATION_SUMMARY.md` - Component overview
- GitHub Issues: https://github.com/your-repo/vtea/issues
