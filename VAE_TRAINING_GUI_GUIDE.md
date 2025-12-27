# VAE Training GUI Guide

Complete guide to the graphical user interface for training 3D Variational Autoencoder models in VTEA.

## Overview

The VAE Training GUI provides a user-friendly interface for training deep learning models without writing code. It combines configuration, data selection, and real-time progress monitoring in both standalone and embedded modes.

### Components

| Component | Purpose | Location |
|-----------|---------|----------|
| **VAETrainingDialog** | Standalone training window | Main entry point |
| **VAETrainingPanel** | Embeddable panel component | For VTEA integration |
| **VAEConfigurationPanel** | Model/hyperparameter settings | Config tab |
| **VAEDataSelectionPanel** | Data source selection | Data tab |
| **VAETrainingProgressPanel** | Real-time visualization | Progress tab |

---

## Launching the Training GUI

### Option 1: Standalone Dialog

Launch the full training dialog as a standalone window:

```java
// From Java code
import vtea.deeplearning.ui.VAETrainingDialog;

VAETrainingDialog dialog = new VAETrainingDialog();
dialog.showDialog();
```

Or run the main method directly:
```bash
java -cp vtea.jar vtea.deeplearning.ui.VAETrainingDialog
```

### Option 2: Embedded Panel

Embed the training interface into existing VTEA components:

```java
// Compact embedded view (just launch button)
VAETrainingPanel panel = new VAETrainingPanel(true);
parentContainer.add(panel);

// Full embedded view (all controls)
VAETrainingPanel panel = new VAETrainingPanel(false);
parentContainer.add(panel);
```

### Option 3: Create Standalone Frame

```java
JFrame frame = VAETrainingPanel.createFrame();
frame.setVisible(true);
```

---

## Configuration Tab

### Model Architecture

**Architecture Presets:**
- **SMALL (32³, 16D)**: Fast training, low memory
  - Input: 32×32×32 voxels
  - Latent: 16 dimensions
  - Channels: [16, 32, 64, 128]
  - Use for: Small cells, testing, limited GPU

- **MEDIUM (64³, 32D)**: Balanced performance (default)
  - Input: 64×64×64 voxels
  - Latent: 32 dimensions
  - Channels: [32, 64, 128, 256]
  - Use for: Most applications

- **LARGE (128³, 64D)**: High detail, slow training
  - Input: 128×128×128 voxels
  - Latent: 64 dimensions
  - Channels: [64, 128, 256, 512]
  - Use for: Large cells, fine details, powerful GPU

- **CUSTOM**: Manual configuration
  - Set all parameters manually
  - Requires deep learning expertise

**Custom Architecture Settings** (enabled only for CUSTOM):
- **Input Size**: Cubic region size in voxels (16-256, must be divisible by 8)
- **Latent Dimension**: Size of learned representation (4-256)
- **Encoder Channels**: Comma-separated channel counts (e.g., "32,64,128,256")
- **Decoder Channels**: Usually reverse of encoder (e.g., "256,128,64,32")

### Training Parameters

**Batch Size** (1-128, default: 8)
- Number of cells processed simultaneously
- Larger = faster but more GPU memory
- Reduce if you get out-of-memory errors
- Typical values:
  - 32³: 16-32
  - 64³: 4-8
  - 128³: 1-2

**Epochs** (1-1000, default: 100)
- Number of complete passes through training data
- More epochs = better learning (up to a point)
- Early stopping prevents overfitting
- Typical values: 50-200

**Learning Rate** (0.00001-0.1, default: 0.001)
- Step size for model updates
- Too high = unstable training
- Too low = slow convergence
- Use 0.001 for most cases
- Reduce to 0.0001 if training diverges

**Use GPU** (checkbox, default: checked)
- Enable CUDA GPU acceleration
- Unchecked = CPU training (very slow)
- Requires NVIDIA GPU with CUDA support
- Check GPU availability before unchecking

### VAE Settings

**Beta (KL weight)** (0.1-10.0, default: 1.0)
- Weight for KL divergence term
- 1.0 = standard VAE
- <1.0 = emphasize reconstruction (sharper outputs)
- >1.0 = β-VAE, emphasize disentanglement
- Start with 1.0, tune if needed

**KL Warmup Epochs** (0-100, default: 10)
- Gradually increase KL weight from 0 to beta
- Prevents posterior collapse
- 0 = no warmup (not recommended)
- 10-20 = typical range
- Increase if latent codes collapse to zero

**Reconstruction Loss** (MSE/BCE/L1, default: MSE)
- **MSE (Mean Squared Error)**: Best for intensity images
- **BCE (Binary Cross Entropy)**: For binary/segmented images
- **L1 (Mean Absolute Error)**: Robust to outliers
- Use MSE unless you have a specific reason

### Output Settings

**Checkpoint Directory**
- Where to save trained models
- Default: ~/vae_checkpoints
- Creates subdirectories for each training run
- Contains:
  - `model.pt`: Model weights
  - `config.json`: Configuration
  - `metadata.json`: Training info
  - `metrics.csv`: Loss curves

---

## Data Tab

### Image Data Source

**Option 1: Use Open ImageJ Images** (default)
- Select from currently open ImageJ windows
- Dropdown shows all open image stacks
- Real-time: opens/closes reflected immediately
- Requires 3D stack (not 2D images)

**Option 2: Use Image File**
- Browse to TIFF file on disk
- Supports multi-page TIFF stacks
- File remains on disk (not loaded into ImageJ)
- Use for very large datasets

### Segmented Objects

**Use Current Segmented Objects** (default)
- Uses objects from MicroBlockSetup
- From VTEA segmentation workflow
- Object count displayed and updated automatically

**Use All Available Objects**
- Future feature: load from database or file
- Currently same as "Current"

**Objects Available**
- Shows count of available cells
- Updates when segmentation changes
- Minimum recommended: 100 cells
- Ideal: 1000+ cells

### Data Options

**Train/Val Split (%)**
- Percentage for training vs. validation
- Default: 80% train, 20% validation
- Range: 50-95%
- Recommendations:
  - Large dataset (>1000): 80/20 or 90/10
  - Small dataset (<500): 70/30
  - Very small (<100): Consider cross-validation instead

**Use Data Augmentation** (checkbox, default: checked)
- Randomly transform training data
- Increases effective dataset size
- Augmentations applied:
  - 90°/180°/270° rotations
  - X/Y/Z flips
  - Small random crops
- Always enabled for training, disabled for validation

---

## Progress Tab

### Training Progress

**Status**: Current training state
- "Ready" - Idle
- "Initializing..." - Loading data
- "Training epoch X/Y" - Active training
- "Training complete!" - Finished
- "Stopped" - User interrupted

**Epoch Progress Bar**
- Overall progress through all epochs
- Shows "Epoch: X / Y"
- Updates after each epoch completes

**Batch Progress Bar**
- Progress within current epoch
- Shows "X / Y" batches
- Updates during epoch

### Loss Charts

**Left Chart: Total Loss**
- **Blue line**: Training loss
- **Red line**: Validation loss
- Lower is better
- Good training:
  - Both lines decreasing
  - Val loss slightly higher than train
  - Lines converging

**Right Chart: Loss Components**
- **Green line**: ELBO (Evidence Lower Bound)
  - Higher is better (negative of total loss)
  - Optimization objective
- **Orange line**: Reconstruction Loss
  - How well model reconstructs images
  - Should decrease steadily
- **Magenta line**: KL Divergence Loss
  - Regularization term
  - Increases during warmup, then stabilizes

### Current Metrics

**Train Loss**: Training set loss (latest epoch)
**Val Loss**: Validation set loss (latest epoch)
**ELBO**: Evidence Lower Bound (higher = better)
**Recon Loss**: Reconstruction component
**KL Loss**: KL divergence component
**Best Val Loss**: Lowest validation loss achieved (⭐ indicates current best)

---

## Control Buttons

### Start Training
- **Color**: Green
- **Action**: Begins training process
- **Enabled**: When not training
- **Behavior**:
  - Validates data availability
  - Switches to Progress tab
  - Starts background training thread
  - Disables configuration tabs

### Stop Training
- **Color**: Red
- **Action**: Stops training gracefully
- **Enabled**: Only during training
- **Behavior**:
  - Signals training to stop
  - Waits for current batch to finish
  - Saves checkpoint at current epoch
  - Re-enables controls

### Load Config
- **Action**: Load configuration from JSON file
- **File format**: `config.json` from previous training
- **Effect**: Populates all configuration fields
- **Enabled**: When not training

### Save Config
- **Action**: Save current configuration to JSON file
- **Default name**: `vae_config.json`
- **Use case**: Save settings for later reuse
- **Enabled**: When not training

### Close
- **Action**: Close training dialog
- **Safety**: Confirms if training in progress
- **Effect**: Stops training and closes window

---

## Training Workflow

### Step-by-Step Guide

**1. Prepare Data**
```
- Open 3D image stack in ImageJ
- Segment cells using VTEA segmentation workflow
- Verify objects appear in MicroBlockSetup
```

**2. Configure Model**
```
- Open VAE Training dialog
- Go to Configuration tab
- Select architecture preset (MEDIUM recommended)
- Set epochs (100 for initial training)
- Check GPU is enabled
- Set checkpoint directory
```

**3. Select Data**
```
- Go to Data tab
- Verify image stack is selected
- Check object count (minimum 100)
- Set train/val split (80% default)
- Enable augmentation
```

**4. Start Training**
```
- Click "Start Training"
- Dialog switches to Progress tab
- Monitor loss curves
- Wait for completion (or stop early)
```

**5. Monitor Progress**
```
- Watch loss curves decrease
- Check validation loss doesn't increase (overfitting)
- Best model auto-saved when val loss improves
- Training stops automatically if early stopping triggered
```

**6. Use Trained Model**
```
- Model saved in checkpoint directory
- Use in FeatureProcessing plugins:
  - VAEFeatureExtraction
  - VAEDimensionalityReduction
  - VAEClustering
  - VAEAnomalyDetection
```

---

## Troubleshooting

### "No image selected or image not available"

**Problem**: No valid 3D image stack found

**Solutions**:
- Open a 3D TIFF stack in ImageJ
- Click "Refresh" in Data tab (if available)
- Verify image has multiple slices (not 2D)
- Try "Use Image File" option instead

### "No segmented objects available"

**Problem**: No cells/objects to train on

**Solutions**:
- Run VTEA segmentation workflow first
- Check MicroBlockSetup has objects loaded
- Verify segmentation completed successfully
- Need minimum ~100 objects for training

### Out of Memory Error

**Problem**: GPU or system RAM exhausted

**Solutions**:
- Reduce batch size (8 → 4 → 2)
- Use smaller architecture (LARGE → MEDIUM → SMALL)
- Close other applications
- Reduce input size (128 → 64 → 32)
- If on CPU, enable GPU instead

### Loss Not Decreasing

**Problem**: Training loss stays high or increases

**Solutions**:
- Check learning rate (try 0.001)
- Increase epochs (100 → 200)
- Verify data quality (not all black/white)
- Check normalization (should be automatic)
- Try different architecture

### Validation Loss Increasing

**Problem**: Overfitting - model memorizing training data

**Solutions**:
- Training will auto-stop (early stopping)
- Increase KL warmup (10 → 20 epochs)
- Increase beta (1.0 → 1.5)
- Add more training data
- Use stronger augmentation
- Reduce model size (LARGE → MEDIUM)

### Training Very Slow

**Problem**: Taking too long per epoch

**Solutions**:
- Enable GPU (must have CUDA GPU)
- Increase batch size (if memory allows)
- Reduce number of objects
- Use smaller architecture
- Check CPU/GPU utilization

### KL Divergence is Zero

**Problem**: Posterior collapse - latent codes not used

**Solutions**:
- Increase KL warmup epochs (10 → 30)
- Decrease beta (1.0 → 0.5)
- Reduce learning rate (0.001 → 0.0005)
- Check reconstruction loss is decreasing
- May resolve naturally after warmup

---

## Advanced Usage

### Resuming Training

**Not yet implemented** - planned feature:
```
- Load checkpoint from disk
- Continue training from saved epoch
- Preserve optimizer state
```

Current workaround:
- Start new training with same config
- Will learn from scratch

### Transfer Learning

Use a pre-trained model for new data:

1. Train base model on large dataset
2. Save checkpoint
3. Load model in code:
```java
VariationalAutoencoder3D model = VariationalAutoencoder3D.load(checkpointPath);
// Fine-tune on new data with lower learning rate
```

### Custom Training Loops

For advanced users, use VAETrainer API directly:
```java
VAETrainer trainer = new VAETrainer(model, config, progressListener);
TrainingResult result = trainer.train(trainLoader, valLoader);
```

### Multi-Channel Training

**Not yet in GUI** - use code:
```java
// Create multi-channel data loader
ImageStack[] channels = {channel1, channel2, channel3};
DataLoader loader = new DataLoader(objects, channels, ...);
```

---

## File Locations

### Default Checkpoint Directory

**Location**: `~/vae_checkpoints/`

**Structure**:
```
vae_checkpoints/
├── run_2025-12-26_14-30-00/
│   ├── model.pt            # Model weights
│   ├── config.json         # Configuration
│   ├── metadata.json       # Training metadata
│   ├── metrics.csv         # Loss history
│   └── best_model.pt       # Best model (lowest val loss)
├── run_2025-12-26_15-45-00/
│   └── ...
```

### Configuration Files

**Format**: JSON
**Location**: User-specified
**Contents**:
```json
{
  "architecture": "MEDIUM",
  "inputSize": 64,
  "latentDim": 32,
  "batchSize": 8,
  "numEpochs": 100,
  "learningRate": 0.001,
  "beta": 1.0,
  "klWarmupEpochs": 10,
  ...
}
```

---

## Integration with VTEA

### Adding to VTEA Menu

To add training dialog to VTEA main menu:

```java
JMenuItem trainMenuItem = new JMenuItem("Train VAE Model...");
trainMenuItem.addActionListener(e -> {
    VAETrainingDialog dialog = new VAETrainingDialog();
    dialog.showDialog();
});
deepLearningMenu.add(trainMenuItem);
```

### Embedding in Workflow

To embed in VTEA workflow panel:

```java
// Add to feature processing or analysis panel
VAETrainingPanel trainingPanel = new VAETrainingPanel(true);
workflowPanel.add(trainingPanel);
```

### ProgressListener Integration

The GUI implements VTEA's ProgressListener interface:

```java
// Connect to VTEA progress bar
VAETrainingProgressPanel progressPanel = dialog.getProgressPanel();
vteaProgressBar.addProgressListener(progressPanel);
```

---

## Performance Benchmarks

### Training Time Estimates

| Architecture | Objects | Epochs | GPU (RTX 3080) | CPU (16-core) |
|--------------|---------|--------|----------------|---------------|
| SMALL (32³) | 1000 | 100 | ~10 min | ~2 hours |
| MEDIUM (64³) | 1000 | 100 | ~30 min | ~8 hours |
| LARGE (128³) | 1000 | 100 | ~2 hours | ~24 hours |
| MEDIUM (64³) | 5000 | 100 | ~2 hours | ~40 hours |

**Recommendation**: Use GPU for all training

### Memory Requirements

| Architecture | Batch Size | GPU Memory | System RAM |
|--------------|------------|------------|------------|
| SMALL (32³) | 16 | ~2 GB | ~4 GB |
| MEDIUM (64³) | 8 | ~4 GB | ~8 GB |
| MEDIUM (64³) | 4 | ~2 GB | ~6 GB |
| LARGE (128³) | 2 | ~6 GB | ~12 GB |

---

## Best Practices

### Data Preparation

✅ **DO**:
- Use at least 100 cells (1000+ ideal)
- Include diverse cell types and states
- Ensure good segmentation quality
- Verify 3D stacks have proper Z-spacing
- Use consistent imaging conditions

❌ **DON'T**:
- Mix different microscopy modalities
- Include debris or artifacts
- Use 2D images (need 3D stacks)
- Combine vastly different cell sizes
- Use extremely low/high intensities

### Model Configuration

✅ **DO**:
- Start with MEDIUM architecture
- Use default hyperparameters first
- Enable GPU if available
- Save configuration for reproducibility
- Monitor validation loss

❌ **DON'T**:
- Start with LARGE without good reason
- Set batch size too high (OOM errors)
- Disable GPU unnecessarily
- Skip validation split
- Ignore early stopping warnings

### Training Process

✅ **DO**:
- Monitor loss curves during training
- Wait for convergence (loss plateaus)
- Save checkpoints regularly (automatic)
- Use augmentation for small datasets
- Validate on held-out data

❌ **DON'T**:
- Stop training too early (<50 epochs)
- Ignore increasing validation loss
- Train without validation split
- Overfit (val loss >> train loss)
- Use same data for train and validation

---

## FAQ

**Q: How many cells do I need for training?**
A: Minimum 100, recommended 1000+. More is better for generalization.

**Q: How long does training take?**
A: 10 minutes to 2 hours on GPU (MEDIUM, 1000 cells, 100 epochs). Much slower on CPU.

**Q: Can I stop and resume training?**
A: Currently no automatic resume. You can stop anytime and start fresh. Best model is saved.

**Q: What if I get out of memory errors?**
A: Reduce batch size, use smaller architecture, or enable GPU if using CPU.

**Q: How do I know when training is done?**
A: Loss plateaus, early stopping triggers, or reaches max epochs. Check Progress tab.

**Q: Can I train on 2D images?**
A: No, VAE requires 3D volumetric data. Convert 2D to single-slice 3D if needed.

**Q: What's the best architecture?**
A: MEDIUM (64³, 32D) for most cases. SMALL for testing, LARGE for large cells with fine detail.

**Q: Should I use data augmentation?**
A: Yes, almost always. Increases effective dataset size and improves generalization.

**Q: How do I use the trained model?**
A: Use in FeatureProcessing plugins (VAEFeatureExtraction, VAEClustering, etc.). Select checkpoint directory.

**Q: Can I train on multiple channels?**
A: Not yet in GUI. Use code API with multi-channel DataLoader.

---

## Getting Help

### Log Files

Check console output for detailed logs:
```
INFO: Starting VAE training...
INFO: Train objects: 800, Val objects: 200
INFO: Epoch 1/100: train_loss=0.1234, val_loss=0.1456
...
```

### Support Resources

- **Documentation**: See `TRAINING_GUIDE.md` for programmatic API
- **Examples**: See `IMPLEMENTATION_SUMMARY.md` for code examples
- **Architecture**: See `VAE_3D_IMPLEMENTATION_PLAN.md` for technical details
- **Plugins**: See `VAE_PLUGIN_GUIDE.md` for using trained models

### Reporting Issues

Include:
- Configuration (save and attach config.json)
- Error messages from console
- System info (OS, GPU, RAM)
- Dataset size and characteristics
- Screenshots of GUI

---

## Summary

The VAE Training GUI provides a complete, user-friendly interface for training deep learning models in VTEA:

✅ **No coding required** - configure, train, and monitor through GUI
✅ **Real-time visualization** - live loss curves and metrics
✅ **Flexible deployment** - standalone dialog or embedded panel
✅ **Production-ready** - comprehensive error handling and validation
✅ **VTEA-integrated** - seamless workflow with FeatureProcessing plugins

Ready to train your first VAE model? Launch the dialog and follow the step-by-step guide above!
