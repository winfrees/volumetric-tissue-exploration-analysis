# Merge Conflict Resolution Strategy

## Problem

The merge between our VAE implementation and upstream has conflicts in 5 files:
1. `pom.xml`
2. `CellRegionExtractor.java`
3. `TensorConverter.java`
4. `AbstractDeepLearningModel.java`
5. `ModelCheckpoint.java`

These files were independently created in both branches with different implementations.

## Analysis

### CellRegionExtractor.java

**Our Version (VAE)**:
```java
// Instance-based API
public class CellRegionExtractor {
    private final int regionSize;
    private final PaddingType paddingType;

    public CellRegionExtractor(int regionSize, PaddingType paddingType) {...}
    public ImageStack extractRegion(MicroObject cell, ImageStack imageStack) {...}
}

// Usage:
CellRegionExtractor extractor = new CellRegionExtractor(64, PaddingType.REPLICATE);
ImageStack region = extractor.extractRegion(cell, imageStack);
```

**Upstream Version (Classification)**:
```java
// Static utility methods
public class CellRegionExtractor {
    public static ImageStack[] extractRegion(MicroObject object, ImageStack[] imageStacks,
                                            int[] regionSize, int[] channels,
                                            PaddingType paddingType) {...}
}

// Usage:
ImageStack[] regions = CellRegionExtractor.extractRegion(
    object, imageStacks, new int[]{64,64,64}, null, PaddingType.REPLICATE);
```

**Key Differences:**
- Ours: Single-channel, instance methods, simple API for VAE
- Upstream: Multi-channel, static methods, flexible for classification
- Ours: 313 lines
- Upstream: 328 lines

**Resolution Strategy:**
✅ **Use upstream as base + add our instance wrapper**

Create unified class:
1. Keep upstream static methods (support classification)
2. Add our instance-based wrapper (support VAE)
3. Both APIs coexist

### TensorConverter.java

**Our Version**:
- Instance-based with NormalizationType enum
- Methods: `imageStackToTensor()`, `imageStacksToTensor()`
- 338 lines

**Upstream Version**:
- Different API structure
- Likely similar functionality

**Resolution:** Merge both APIs into one class

### AbstractDeepLearningModel.java

**Our Version**:
- VAE-specific base class
- No class definitions (unsupervised)
- 226 lines

**Upstream Version**:
- Classification-specific base class
- Has ClassDefinition support
- Different structure

**Resolution:** These should potentially be SEPARATE classes:
- `AbstractDeepLearningModel` (upstream, for classification)
- `AbstractVAEModel` (ours, for VAE)

### ModelCheckpoint.java

**Our Version**:
- VAE-specific checkpoint format
- Saves model.pt, config.json, metadata.json, metrics.csv
- 361 lines

**Upstream Version**:
- Classification checkpoint format
- Different structure

**Resolution:** Similar to above - might need separate classes or unified format

### pom.xml

**Both versions** added PyTorch dependencies.

**Resolution:** Merge dependencies, ensure no duplicates

## Recommended Approach

### Strategy 1: Unified Classes with Dual APIs (Recommended for Shared Utilities)

For `CellRegionExtractor` and `TensorConverter`:
```java
public class CellRegionExtractor {
    // === STATIC API (Classification) ===
    public static ImageStack[] extractRegion(...) { ... }

    // === INSTANCE API (VAE) ===
    private final int regionSize;
    private final PaddingType paddingType;

    public CellRegionExtractor(int regionSize, PaddingType paddingType) { ... }

    public ImageStack extractRegion(MicroObject cell, ImageStack stack) {
        // Delegate to static method
        ImageStack[] result = extractRegion(
            cell, new ImageStack[]{stack},
            new int[]{regionSize, regionSize, regionSize},
            null, paddingType
        );
        return result[0];
    }
}
```

### Strategy 2: Separate Classes (Recommended for Model-Specific Code)

For `AbstractDeepLearningModel` and `ModelCheckpoint`:
```
AbstractDeepLearningModel.java    (upstream, classification)
AbstractVAEModel.java              (ours, VAE-specific)

ModelCheckpoint.java               (upstream, classification)
VAEModelCheckpoint.java            (ours, VAE-specific)
```

## Implementation Plan

### Step 1: Accept Upstream Versions

For conflicted files, take upstream version as base:
```bash
git checkout --theirs pom.xml
git checkout --theirs src/main/java/vtea/deeplearning/data/CellRegionExtractor.java
git checkout --theirs src/main/java/vtea/deeplearning/data/TensorConverter.java
git checkout --theirs src/main/java/vtea/deeplearning/models/AbstractDeepLearningModel.java
git checkout --theirs src/main/java/vtea/deeplearning/training/ModelCheckpoint.java
```

### Step 2: Create VAE-Specific Versions

Rename our versions to be VAE-specific:
```bash
# Our files become VAE-specific
mv AbstractDeepLearningModel.java → Keep upstream, create VAEModel separately
mv ModelCheckpoint.java → Keep upstream, create VAEModelCheckpoint.java
```

### Step 3: Add Wrapper Methods to Shared Utilities

Add our instance-based API to upstream utilities:
- `CellRegionExtractor`: Add constructor + instance methods
- `TensorConverter`: Merge both APIs

### Step 4: Update VAE Code to Use New APIs

Update all VAE files:
- `VAEDataLoader.java`
- `VAETrainer.java`
- `VAE*.java` plugins
- UI components

## Detailed Resolution for Each File

### CellRegionExtractor.java

```java
/*
 * Unified version supporting both Classification and VAE
 */
package vtea.deeplearning.data;

import ij.ImageStack;
import ij.process.ImageProcessor;
import ij.process.FloatProcessor;
import vteaobjects.MicroObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for extracting 3D cubic regions around segmented cells.
 *
 * Supports two APIs:
 * 1. Static methods for multi-channel extraction (classification)
 * 2. Instance methods for single-channel extraction (VAE)
 *
 * @author VTEA Deep Learning Team
 */
public class CellRegionExtractor {

    private static final Logger logger = LoggerFactory.getLogger(CellRegionExtractor.class);

    // Instance fields for VAE API
    private final int regionSize;
    private final PaddingType paddingType;

    public enum PaddingType {
        ZERO,       // Pad with zeros
        MIRROR,     // Mirror the edge pixels
        REPLICATE   // Replicate the edge pixels
    }

    // ===== STATIC API (for Classification) =====

    public static ImageStack[] extractRegion(MicroObject object, ImageStack[] imageStacks,
                                            int[] regionSize, int[] channels,
                                            PaddingType paddingType) {
        // Upstream implementation here
        ...
    }

    // ===== INSTANCE API (for VAE) =====

    public CellRegionExtractor(int regionSize, PaddingType paddingType) {
        this.regionSize = regionSize;
        this.paddingType = paddingType;
        logger.debug("Created CellRegionExtractor: size={}, padding={}", regionSize, paddingType);
    }

    public ImageStack extractRegion(MicroObject cell, ImageStack imageStack) {
        // Delegate to static method
        ImageStack[] result = extractRegion(
            cell,
            new ImageStack[]{imageStack},
            new int[]{regionSize, regionSize, regionSize},
            null,
            paddingType
        );
        return result[0];
    }

    public ImageStack[] extractRegion(MicroObject cell, ImageStack[] imageStacks) {
        return extractRegion(
            cell,
            imageStacks,
            new int[]{regionSize, regionSize, regionSize},
            null,
            paddingType
        );
    }
}
```

### TensorConverter.java

Similar approach - merge both APIs.

### AbstractDeepLearningModel.java

Keep upstream version as-is. Our VAE models don't actually extend it currently, so no changes needed.

### ModelCheckpoint.java

Rename ours to `VAEModelCheckpoint.java` to avoid conflict.

### pom.xml

Merge dependencies carefully.

## Next Steps

1. Accept upstream versions
2. Create VAE-specific renamed versions where needed
3. Add wrapper methods to shared utilities
4. Update VAE code references
5. Test compilation
6. Commit resolution

This approach maintains compatibility with both systems while avoiding conflicts.
