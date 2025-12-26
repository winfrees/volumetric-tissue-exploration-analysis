# Implementation Plan: 3D Deep Learning Classification for VTEA

**Version:** 1.0
**Date:** 2025-12-26
**Branch:** `claude/3d-image-classification-ncERT`

---

## Table of Contents

1. [Current VTEA Architecture Understanding](#current-architecture)
2. [NephNet3D Architecture](#nephnet3d-architecture)
3. [Implementation Plan Overview](#implementation-overview)
4. [Phase 1: Core Infrastructure Setup](#phase-1)
5. [Phase 2: Data Pipeline](#phase-2)
6. [Phase 3: User-Defined Class Naming System](#phase-3)
7. [Phase 4: Model Architecture](#phase-4)
8. [Phase 5: Training Infrastructure](#phase-5)
9. [Phase 6: Integration with VTEA](#phase-6)
10. [Phase 7: UI Components](#phase-7)
11. [Implementation Sequence](#implementation-sequence)
12. [Potential Challenges & Solutions](#challenges)
13. [Missing Components / Questions](#questions)
14. [Testing Strategy](#testing)
15. [Documentation Needs](#documentation)

---

<a name="current-architecture"></a>
## 1. Current VTEA Architecture Understanding

### Key Components

**FeatureProcessing Framework** (`vtea.featureprocessing`)
- Plugin-based system for clustering/classification
- `AbstractFeatureProcessing`: Base class with data normalization, column selection utilities
- Current implementations: K-means, Gaussian Mixture, Hierarchical clustering, t-SNE, PCA
- All features extend `AbstractFeatureProcessing` and implement `FeatureProcessing` interface

**Object Representation** (`vteaobjects.MicroObject`)
- Segmented cells stored as:
  - 3D pixel coordinates: `int[] x, y, z`
  - Features: `ArrayList<ArrayList<Number>>`
  - Centroids, measurements, morphological data
- Serializable for persistence

**Image Handling**
- ImageJ's `ImageStack` for 3D volumetric data
- Multi-channel support via `ImageStack[]`
- Integration with ImageJ ecosystem

**Classification/Labeling**
- Manual classification UI exists (`ManualClassification.java`)
- Currently supports numeric class assignment
- Integration with MicroExplorer for visualization

---

<a name="nephnet3d-architecture"></a>
## 2. NephNet3D Architecture

Based on research from the published paper (Cytometry Part A, 2020):

### Architecture Details

**Input:**
- Single-channel 3D volumes (extensible to n-channels)
- Typical size: variable based on cell region extraction

**Convolutional Blocks** (4 total):
1. **Conv Block 1:** Input channels → 32 features
   - Conv3d (3×3×3, stride=2, padding=1) → BatchNorm3d → LeakyReLU
   - Conv3d (3×3×3, padding=1) → LeakyReLU
   - MaxPool3d (stride=2)

2. **Conv Block 2:** 32 → 64 features (same structure)

3. **Conv Block 3:** 64 → 128 features (same structure)

4. **Conv Block 4:** 128 → 256 features
   - Conv3d (kernel=(1,3,3), padding=(0,1,1)) → BatchNorm3d → LeakyReLU

**Fully Connected Classifier:**
- FC5: 256×4×4 → 256 features, LeakyReLU, BatchNorm1d, Dropout(0.5)
- FC6: 256 → 128 features, LeakyReLU, BatchNorm1d, Dropout(0.5)
- FC7: 128 → num_classes (output)

**Weight Initialization:**
- Kaiming normal for Conv3d and Linear layers
- Normal distribution for biases

**Training Configuration:**
- Loss: CrossEntropyLoss
- Optimizer: Adam or SGD with momentum
- Balanced accuracy metric for evaluation
- Support for 8 cell types in kidney tissue

---

<a name="implementation-overview"></a>
## 3. Implementation Plan Overview

### Goals
1. Integrate bytedeco PyTorch bindings into VTEA
2. Implement generic 3D image classification framework
3. Implement NephNet3D architecture specifically
4. Support user-defined class naming for interpretability
5. Enable training on manually labeled cell regions
6. Integrate inference into existing VTEA workflow
7. Maintain compatibility with existing clustering/classification plugins

### Package Structure
```
vtea/
├── deeplearning/
│   ├── models/
│   │   ├── AbstractDeepLearningModel.java
│   │   ├── NephNet3D.java
│   │   └── Generic3DCNN.java
│   ├── data/
│   │   ├── TensorConverter.java
│   │   ├── DatasetDefinition.java
│   │   ├── ClassDefinition.java
│   │   └── CellRegionExtractor.java
│   ├── training/
│   │   ├── Trainer.java
│   │   ├── DataLoader.java
│   │   └── ModelCheckpoint.java
│   └── inference/
│       └── DeepLearningClassification.java
├── exploration/
│   └── plottools/
│       └── panels/
│           └── ManualClassification.java (enhanced)
```

---

<a name="phase-1"></a>
## 4. Phase 1: Core Infrastructure Setup

### 4.1 Dependency Management

**Add to `pom.xml`:**
```xml
<!-- PyTorch JavaCPP Bindings -->
<dependency>
    <groupId>org.bytedeco</groupId>
    <artifactId>pytorch-platform</artifactId>
    <version>2.9.1-1.5.11</version>
</dependency>

<!-- Optional: GPU Support -->
<dependency>
    <groupId>org.bytedeco</groupId>
    <artifactId>cuda-platform</artifactId>
    <version>12.3-8.9-1.5.11</version>
    <optional>true</optional>
</dependency>

<!-- JSON for class definitions -->
<dependency>
    <groupId>com.google.code.gson</groupId>
    <artifactId>gson</artifactId>
    <version>2.10.1</version>
</dependency>
```

### 4.2 Package Creation

Create directory structure:
```
src/main/java/vtea/deeplearning/
├── models/
├── data/
├── training/
└── inference/
```

### 4.3 Configuration Management

**Create `DeepLearningConfig.java`:**
- Device selection (CPU/GPU)
- Default model paths
- Default hyperparameters
- Memory management settings

---

<a name="phase-2"></a>
## 5. Phase 2: Data Pipeline

### 5.1 Tensor Conversion Utilities

**File:** `vtea/deeplearning/data/TensorConverter.java`

**Responsibilities:**
- Convert `ImageStack` to PyTorch `Tensor` (float32)
- Support n-channel conversion from `ImageStack[]`
- Handle normalization strategies:
  - Z-score normalization (mean=0, std=1)
  - Min-max normalization (0-1 range)
  - Per-channel normalization
- Reverse conversion `Tensor` → `ImageStack` for visualization
- Batch tensor creation from multiple regions

**Key Methods:**
```java
public class TensorConverter {
    public static Tensor imageStackToTensor(ImageStack stack, boolean normalize);
    public static Tensor multiChannelToTensor(ImageStack[] stacks, int[] channelIndices);
    public static ImageStack tensorToImageStack(Tensor tensor);
    public static Tensor batchRegionsToTensor(List<ImageStack[]> regions);
    public static Tensor normalizeZScore(Tensor input);
    public static Tensor normalizeMinMax(Tensor input);
}
```

### 5.2 Cell Region Extraction

**File:** `vtea/deeplearning/data/CellRegionExtractor.java`

**Responsibilities:**
- Extract 3D cubic regions centered on `MicroObject` centroids
- Configurable region size (e.g., 32×32×32, 64×64×64, 128×128×128)
- Handle boundary cases with padding strategies:
  - Zero padding
  - Mirror padding
  - Replicate padding
- Multi-channel extraction from `ImageStack[]`
- Efficient extraction using bounding box optimization

**Key Methods:**
```java
public class CellRegionExtractor {
    public ImageStack[] extractRegion(MicroObject obj, ImageStack[] fullImage,
                                       int[] regionSize, int[] channels);
    public List<ImageStack[]> extractBatch(List<MicroObject> objects,
                                            ImageStack[] fullImage,
                                            DatasetDefinition config);
    public ImageStack applyPadding(ImageStack region, int[] targetSize,
                                   PaddingType type);
}
```

### 5.3 Dataset Definition

**File:** `vtea/deeplearning/data/DatasetDefinition.java`

**Responsibilities:**
- Configuration for region extraction and preprocessing
- Manages class definitions (see Phase 3)
- Serializable for persistence

**Implementation:**
```java
public class DatasetDefinition implements Serializable {
    // Region extraction
    private int[] regionSize;           // [depth, height, width]
    private int[] channels;             // channel indices to extract
    private PaddingType paddingType;    // boundary handling
    private int padding;                // padding amount

    // Preprocessing
    private boolean normalize;          // apply normalization
    private NormalizationType normType; // "zscore", "minmax", "none"

    // Class definitions (Phase 3)
    private HashMap<Integer, ClassDefinition> classDefinitions;

    // Metadata
    private String name;
    private String description;
    private Date created;

    // Methods
    public void addClassDefinition(ClassDefinition classDef);
    public ClassDefinition getClassDefinition(int classId);
    public String getClassName(int classId);
    public void setClassName(int classId, String name);
    public List<String> getAllClassNames();
    public int getNumClasses();
}

public enum PaddingType { ZERO, MIRROR, REPLICATE }
public enum NormalizationType { ZSCORE, MINMAX, NONE }
```

---

<a name="phase-3"></a>
## 6. Phase 3: User-Defined Class Naming System

### 6.1 Overview

Enable users to assign meaningful, human-readable names to cell classification classes instead of auto-generated labels (e.g., "Podocyte" instead of "Cluster_0"). This enhances interpretability and scientific communication.

### 6.2 Class Definition Data Structure

**File:** `vtea/deeplearning/data/ClassDefinition.java`

**Implementation:**
```java
public class ClassDefinition implements Serializable {
    private int classId;                // Numeric identifier (0, 1, 2, ...)
    private String className;           // User-provided name (e.g., "Podocyte")
    private String description;         // Detailed description (optional)
    private Color displayColor;         // Visualization color

    // Metadata
    private int sampleCount;            // Number of labeled samples
    private Date created;               // Creation timestamp
    private Date lastModified;          // Last update timestamp
    private String author;              // User who created this class

    // Constructors
    public ClassDefinition(int classId, String className) {
        this.classId = classId;
        this.className = className;
        this.displayColor = generateRandomColor();
        this.created = new Date();
    }

    public ClassDefinition(int classId, String className, Color color) {
        this(classId, className);
        this.displayColor = color;
    }

    // Methods
    public void incrementSampleCount() { this.sampleCount++; }
    public void setSampleCount(int count) { this.sampleCount = count; }

    // JSON serialization helpers
    public String toJson();
    public static ClassDefinition fromJson(String json);

    // Validation
    public boolean isValid() {
        return classId >= 0 && className != null && !className.isEmpty();
    }

    private Color generateRandomColor() {
        // Generate visually distinct colors
        Random rand = new Random(classId);
        return new Color(rand.nextInt(256), rand.nextInt(256), rand.nextInt(256));
    }
}
```

### 6.3 Enhanced Manual Classification UI

**File:** `vtea/exploration/plottools/panels/ManualClassification.java` (enhanced)

**New Features:**
1. **Class Name Input:** Text fields for custom class names
2. **Color Selection:** Color picker for each class
3. **Class Management:** Add/remove/edit classes dynamically
4. **Class List Display:** Show all defined classes with counts
5. **Import/Export:** Load class definitions from previous sessions

**UI Components:**

```java
public class ManualClassification implements WindowListener {

    // Enhanced fields
    private ArrayList<ClassDefinition> classDefinitions = new ArrayList<>();
    private HashMap<Integer, JTextField> classNameFields = new HashMap<>();
    private HashMap<Integer, JButton> colorPickerButtons = new HashMap<>();
    private JPanel classDefinitionPanel;

    // New methods
    private void setupClassDefinitionPanel() {
        classDefinitionPanel = new JPanel(new GridBagLayout());

        for (int i = 0; i < nClasses; i++) {
            // Class name field
            JLabel label = new JLabel("Class " + i + " Name:");
            JTextField nameField = new JTextField("Class_" + i, 20);
            classNameFields.put(i, nameField);

            // Color picker button
            JButton colorButton = new JButton("Color");
            colorButton.setBackground(generateColor(i));
            colorPickerButtons.put(i, colorButton);
            colorButton.addActionListener(e -> showColorPicker(i));

            // Add to panel
            classDefinitionPanel.add(label);
            classDefinitionPanel.add(nameField);
            classDefinitionPanel.add(colorButton);
        }
    }

    private void showColorPicker(int classId) {
        Color newColor = JColorChooser.showDialog(
            classLoggerFrame,
            "Choose Color for " + classNameFields.get(classId).getText(),
            colorPickerButtons.get(classId).getBackground()
        );
        if (newColor != null) {
            colorPickerButtons.get(classId).setBackground(newColor);
            classDefinitions.get(classId).setDisplayColor(newColor);
        }
    }

    private void buildClassDefinitions() {
        classDefinitions.clear();
        for (int i = 0; i < nClasses; i++) {
            String name = classNameFields.get(i).getText().trim();
            if (name.isEmpty()) name = "Class_" + i;

            Color color = colorPickerButtons.get(i).getBackground();
            ClassDefinition classDef = new ClassDefinition(i, name, color);
            classDefinitions.add(classDef);
        }
    }

    private void saveClassDefinitions(String path) {
        // Save to JSON file
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (FileWriter writer = new FileWriter(path)) {
            gson.toJson(classDefinitions, writer);
        } catch (IOException e) {
            IJ.error("Failed to save class definitions: " + e.getMessage());
        }
    }

    private void loadClassDefinitions(String path) {
        // Load from JSON file
        Gson gson = new Gson();
        try (FileReader reader = new FileReader(path)) {
            Type listType = new TypeToken<ArrayList<ClassDefinition>>(){}.getType();
            classDefinitions = gson.fromJson(reader, listType);
            updateUIFromDefinitions();
        } catch (IOException e) {
            IJ.error("Failed to load class definitions: " + e.getMessage());
        }
    }

    private void updateUIFromDefinitions() {
        for (ClassDefinition classDef : classDefinitions) {
            int id = classDef.getClassId();
            if (classNameFields.containsKey(id)) {
                classNameFields.get(id).setText(classDef.getClassName());
                colorPickerButtons.get(id).setBackground(classDef.getDisplayColor());
            }
        }
    }
}
```

**Enhanced Setup Dialog:**

```java
class SetupManualClassification extends JDialog {
    private JSpinner numClassesSpinner;
    private JSpinner numCellsSpinner;
    private JButton importButton;
    private JButton exportButton;

    // Allow importing class definitions from previous sessions
    private void setupImportExport() {
        importButton = new JButton("Import Class Definitions");
        importButton.addActionListener(e -> {
            JFileChooser fc = new JFileChooser();
            fc.setFileFilter(new FileNameExtensionFilter("JSON files", "json"));
            if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                loadClassDefinitionsFromFile(fc.getSelectedFile());
            }
        });

        exportButton = new JButton("Export Class Definitions");
        exportButton.addActionListener(e -> {
            JFileChooser fc = new JFileChooser();
            fc.setFileFilter(new FileNameExtensionFilter("JSON files", "json"));
            if (fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                saveClassDefinitionsToFile(fc.getSelectedFile());
            }
        });
    }
}
```

### 6.4 Model Integration

**Update `AbstractDeepLearningModel.java`:**

```java
public abstract class AbstractDeepLearningModel {
    protected HashMap<Integer, ClassDefinition> classDefinitions;
    protected int numClasses;

    // Class definition management
    public void setClassDefinitions(HashMap<Integer, ClassDefinition> defs) {
        this.classDefinitions = defs;
        this.numClasses = defs.size();
    }

    public HashMap<Integer, ClassDefinition> getClassDefinitions() {
        return classDefinitions;
    }

    public String getClassName(int classId) {
        return classDefinitions.containsKey(classId) ?
               classDefinitions.get(classId).getClassName() :
               "Unknown_" + classId;
    }

    public Color getClassColor(int classId) {
        return classDefinitions.containsKey(classId) ?
               classDefinitions.get(classId).getDisplayColor() :
               Color.GRAY;
    }

    // Save/load with model checkpoints
    public abstract void saveWithClassDefinitions(String modelPath, String classDefPath);
    public abstract void loadWithClassDefinitions(String modelPath, String classDefPath);
}
```

### 6.5 Model Checkpoint Integration

**Update `ModelCheckpoint.java`:**

```java
public class ModelCheckpoint {

    /**
     * Saves model with three files:
     * - model.pt: PyTorch weights
     * - model_classes.json: Class definitions with names/colors
     * - model_config.json: Architecture configuration
     */
    public static void save(AbstractDeepLearningModel model, String basePath) {
        // Save PyTorch weights
        String weightsPath = basePath + ".pt";
        model.save(weightsPath);

        // Save class definitions
        String classDefPath = basePath + "_classes.json";
        saveClassDefinitions(model.getClassDefinitions(), classDefPath);

        // Save architecture config
        String configPath = basePath + "_config.json";
        saveModelConfig(model, configPath);

        IJ.log("Model saved to: " + basePath);
    }

    private static void saveClassDefinitions(
        HashMap<Integer, ClassDefinition> classDefs,
        String path
    ) {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (FileWriter writer = new FileWriter(path)) {
            gson.toJson(classDefs.values(), writer);
            IJ.log("Class definitions saved: " + classDefs.size() + " classes");
        } catch (IOException e) {
            IJ.error("Failed to save class definitions: " + e.getMessage());
        }
    }

    public static AbstractDeepLearningModel load(String basePath, String modelType) {
        // Load model architecture
        AbstractDeepLearningModel model = createModel(modelType, basePath + "_config.json");

        // Load weights
        model.load(basePath + ".pt");

        // Load class definitions
        HashMap<Integer, ClassDefinition> classDefs =
            loadClassDefinitions(basePath + "_classes.json");
        model.setClassDefinitions(classDefs);

        IJ.log("Model loaded with " + classDefs.size() + " class definitions");
        return model;
    }

    private static HashMap<Integer, ClassDefinition> loadClassDefinitions(String path) {
        HashMap<Integer, ClassDefinition> result = new HashMap<>();
        Gson gson = new Gson();

        try (FileReader reader = new FileReader(path)) {
            Type listType = new TypeToken<ArrayList<ClassDefinition>>(){}.getType();
            ArrayList<ClassDefinition> list = gson.fromJson(reader, listType);

            for (ClassDefinition def : list) {
                result.put(def.getClassId(), def);
            }
        } catch (IOException e) {
            IJ.log("Warning: Could not load class definitions, using defaults");
            // Generate default class definitions
            result = generateDefaultClassDefinitions(8); // Default to 8 classes
        }

        return result;
    }

    private static HashMap<Integer, ClassDefinition> generateDefaultClassDefinitions(int numClasses) {
        HashMap<Integer, ClassDefinition> result = new HashMap<>();
        for (int i = 0; i < numClasses; i++) {
            result.put(i, new ClassDefinition(i, "Class_" + i));
        }
        return result;
    }
}
```

### 6.6 JSON Format for Class Definitions

**File Format:** `model_classes.json`

```json
{
  "classes": [
    {
      "classId": 0,
      "className": "Podocyte",
      "description": "Glomerular epithelial cells with foot processes",
      "displayColor": {
        "r": 255,
        "g": 0,
        "b": 0
      },
      "sampleCount": 150,
      "created": "2025-12-26T10:30:00Z",
      "lastModified": "2025-12-26T10:30:00Z",
      "author": "researcher1"
    },
    {
      "classId": 1,
      "className": "Endothelial",
      "description": "Glomerular capillary endothelial cells",
      "displayColor": {
        "r": 0,
        "g": 255,
        "b": 0
      },
      "sampleCount": 200,
      "created": "2025-12-26T10:30:00Z",
      "lastModified": "2025-12-26T10:30:00Z",
      "author": "researcher1"
    },
    {
      "classId": 2,
      "className": "Mesangial",
      "description": "Mesangial cells in glomerulus",
      "displayColor": {
        "r": 0,
        "g": 0,
        "b": 255
      },
      "sampleCount": 120,
      "created": "2025-12-26T10:30:00Z",
      "lastModified": "2025-12-26T10:30:00Z",
      "author": "researcher1"
    }
  ],
  "metadata": {
    "version": "1.0",
    "totalClasses": 3,
    "totalSamples": 470,
    "created": "2025-12-26T10:30:00Z"
  }
}
```

### 6.7 Inference Results with Class Names

**Update `DeepLearningClassification.java`:**

```java
@Plugin(type = FeatureProcessing.class)
public class DeepLearningClassification extends AbstractFeatureProcessing {

    private AbstractDeepLearningModel model;
    private HashMap<Integer, ClassDefinition> classDefinitions;

    @Override
    public boolean process(ArrayList al, double[][] feature, boolean validate) {
        // Extract regions, run inference
        int[] predictions = model.predict(regions);
        float[][] confidences = model.predictProba(regions);

        // Build results with class names
        ArrayList<String> classNames = new ArrayList<>();
        ArrayList<Float> confidenceScores = new ArrayList<>();

        for (int i = 0; i < predictions.length; i++) {
            int predClass = predictions[i];
            String className = model.getClassName(predClass);
            float confidence = confidences[i][predClass];

            classNames.add(className);
            confidenceScores.add(confidence);

            IJ.log(String.format("Object %d: %s (%.2f%%)",
                   i, className, confidence * 100));
        }

        // Store in dataResult for VTEA
        dataResult.clear();
        dataResult.add(classNames);
        dataResult.add(confidenceScores);

        return true;
    }

    @Override
    public String getDataDescription(ArrayList params) {
        String modelName = model.getClass().getSimpleName();
        int numClasses = classDefinitions.size();
        return String.format("%s_Classification_%d_classes_%s",
                           modelName, numClasses, getCurrentTime());
    }

    // Export results to CSV with class names
    public void exportResults(String path, ArrayList<MicroObject> objects) {
        try (PrintWriter writer = new PrintWriter(path)) {
            // Header
            writer.println("ObjectID,PredictedClass,ClassName,Confidence,CentroidX,CentroidY,CentroidZ");

            ArrayList<String> classNames = (ArrayList<String>) dataResult.get(0);
            ArrayList<Float> confidences = (ArrayList<Float>) dataResult.get(1);

            for (int i = 0; i < objects.size(); i++) {
                MicroObject obj = objects.get(i);
                writer.printf("%d,%d,%s,%.4f,%.2f,%.2f,%.2f%n",
                    i,
                    getClassIdFromName(classNames.get(i)),
                    classNames.get(i),
                    confidences.get(i),
                    obj.getCentroidX(),
                    obj.getCentroidY(),
                    obj.getCentroidZ()
                );
            }
        } catch (IOException e) {
            IJ.error("Failed to export results: " + e.getMessage());
        }
    }
}
```

### 6.8 Visualization with Custom Class Names

**MicroExplorer Integration:**

```java
public class MicroExplorerVisualization {

    public void visualizeClassifications(
        ArrayList<MicroObject> objects,
        ArrayList<String> classNames,
        HashMap<Integer, ClassDefinition> classDefinitions
    ) {
        // Create color-coded overlay
        for (int i = 0; i < objects.size(); i++) {
            MicroObject obj = objects.get(i);
            String className = classNames.get(i);

            // Get color from class definition
            ClassDefinition classDef = findClassDefinitionByName(className, classDefinitions);
            Color color = classDef != null ? classDef.getDisplayColor() : Color.GRAY;

            // Color object in visualization
            obj.setColor(color);
        }

        // Create legend
        createLegend(classDefinitions);
    }

    private void createLegend(HashMap<Integer, ClassDefinition> classDefinitions) {
        JPanel legendPanel = new JPanel();
        legendPanel.setLayout(new BoxLayout(legendPanel, BoxLayout.Y_AXIS));

        for (ClassDefinition classDef : classDefinitions.values()) {
            JPanel row = new JPanel(new FlowLayout(FlowLayout.LEFT));

            // Color square
            JPanel colorSquare = new JPanel();
            colorSquare.setBackground(classDef.getDisplayColor());
            colorSquare.setPreferredSize(new Dimension(20, 20));

            // Class name label
            JLabel nameLabel = new JLabel(classDef.getClassName());

            // Sample count
            JLabel countLabel = new JLabel("(n=" + classDef.getSampleCount() + ")");
            countLabel.setForeground(Color.GRAY);

            row.add(colorSquare);
            row.add(nameLabel);
            row.add(countLabel);
            legendPanel.add(row);
        }

        // Display legend window
        JFrame legendFrame = new JFrame("Classification Legend");
        legendFrame.add(legendPanel);
        legendFrame.pack();
        legendFrame.setVisible(true);
    }
}
```

### 6.9 Workflow Examples

**Training Workflow with Custom Classes:**

```
1. User opens VTEA → Manual Classification Tool
2. Setup dialog:
   - Number of classes: 3
   - Cells to label: 50
   - [Import Class Definitions] (optional)

3. Class Definition Panel:
   Class 0 Name: [Podocyte      ] [🎨 Red   ]
   Class 1 Name: [Endothelial   ] [🎨 Green ]
   Class 2 Name: [Mesangial     ] [🎨 Blue  ]

4. User labels cells interactively:
   - Click cell → Assign to class
   - Display shows colored regions

5. Save labeled data:
   - labeled_data.csv (cell IDs + class assignments)
   - class_definitions.json (class metadata)

6. Train model:
   - Load labeled data
   - DatasetDefinition includes class definitions
   - Model trains with class names embedded

7. Save trained model:
   - nephnet3d_kidney.pt (weights)
   - nephnet3d_kidney_classes.json (class names)
   - nephnet3d_kidney_config.json (architecture)
```

**Inference Workflow:**

```
1. Load pretrained model:
   - Automatically loads class definitions
   - UI shows: "Model loaded with 3 classes: Podocyte, Endothelial, Mesangial"

2. Run classification on new image:
   - Extract regions around all cells
   - Batch prediction

3. Results display:
   Object 0001: Podocyte (92% confidence)
   Object 0002: Endothelial (87% confidence)
   Object 0003: Podocyte (95% confidence)
   ...

4. Visualization:
   - Cells colored by class (red, green, blue)
   - Legend shows class names and colors

5. Export results:
   - results.csv with "ClassName" column
   - Color-coded image overlay
```

### 6.10 Backward Compatibility

For existing clustering plugins without custom names:

```java
public class ClusteringBackwardCompatibility {

    /**
     * Generates default class definitions for clustering results
     * that don't have custom names
     */
    public static HashMap<Integer, ClassDefinition> generateDefaultDefinitions(
        int numClusters,
        String prefix
    ) {
        HashMap<Integer, ClassDefinition> result = new HashMap<>();

        for (int i = 0; i < numClusters; i++) {
            String name = prefix + "_" + i;  // e.g., "Cluster_0", "Cluster_1"
            ClassDefinition def = new ClassDefinition(i, name);
            result.put(i, def);
        }

        return result;
    }

    /**
     * Wraps existing clustering results with class definitions
     */
    public static void wrapClusteringResults(
        ArrayList clusterAssignments,
        int numClusters
    ) {
        HashMap<Integer, ClassDefinition> classDefs =
            generateDefaultDefinitions(numClusters, "Cluster");

        // Store definitions for visualization
        storeInSessionContext(classDefs);
    }
}
```

---

<a name="phase-4"></a>
## 7. Phase 4: Model Architecture

### 7.1 Base Architecture

**File:** `vtea/deeplearning/models/AbstractDeepLearningModel.java`

**Responsibilities:**
- Abstract base for all deep learning models
- Common functionality: save/load, device management
- Interface for forward pass, training, inference

**Implementation:**
```java
public abstract class AbstractDeepLearningModel implements Serializable {

    protected String modelName;
    protected int inputChannels;
    protected int numClasses;
    protected int[] inputShape;  // [channels, depth, height, width]
    protected HashMap<Integer, ClassDefinition> classDefinitions;

    // PyTorch components
    protected transient Module network;
    protected transient Device device;

    // Abstract methods
    public abstract Tensor forward(Tensor input);
    public abstract void buildArchitecture();
    public abstract void initializeWeights();

    // Common methods
    public void save(String path) {
        network.save(path);
    }

    public void load(String path) {
        network.load(path);
    }

    public void toDevice(Device device) {
        this.device = device;
        network.to(device);
    }

    public int[] predict(List<Tensor> inputs) {
        network.eval();
        int[] predictions = new int[inputs.size()];

        for (int i = 0; i < inputs.size(); i++) {
            Tensor output = forward(inputs.get(i));
            predictions[i] = argmax(output);
        }

        return predictions;
    }

    public float[][] predictProba(List<Tensor> inputs) {
        network.eval();
        float[][] probabilities = new float[inputs.size()][numClasses];

        for (int i = 0; i < inputs.size(); i++) {
            Tensor output = forward(inputs.get(i));
            Tensor probs = softmax(output);
            probabilities[i] = tensorToFloatArray(probs);
        }

        return probabilities;
    }

    // Utility methods
    protected int argmax(Tensor tensor) {
        // Implementation using PyTorch operations
    }

    protected Tensor softmax(Tensor logits) {
        // Implementation using PyTorch operations
    }
}
```

### 7.2 NephNet3D Implementation

**File:** `vtea/deeplearning/models/NephNet3D.java`

**Implementation:**
```java
public class NephNet3D extends AbstractDeepLearningModel {

    private int baseFeatures;

    // Convolutional blocks
    private Sequential conv1, conv2, conv3, conv4;

    // Classifier
    private Sequential fc5, fc6, fc7;

    public NephNet3D(int inputChannels, int numClasses, int baseFeatures) {
        this.modelName = "NephNet3D";
        this.inputChannels = inputChannels;
        this.numClasses = numClasses;
        this.baseFeatures = baseFeatures;  // Default: 32

        buildArchitecture();
        initializeWeights();
    }

    @Override
    public void buildArchitecture() {
        // Conv Block 1: inputChannels → baseFeatures (32)
        conv1 = createConvBlock(inputChannels, baseFeatures, true);

        // Conv Block 2: baseFeatures → baseFeatures*2 (64)
        conv2 = createConvBlock(baseFeatures, baseFeatures * 2, true);

        // Conv Block 3: baseFeatures*2 → baseFeatures*4 (128)
        conv3 = createConvBlock(baseFeatures * 2, baseFeatures * 4, true);

        // Conv Block 4: baseFeatures*4 → baseFeatures*8 (256)
        conv4 = createConvBlock4(baseFeatures * 4, baseFeatures * 8);

        // Fully connected classifier
        fc5 = Sequential.of(
            Linear.of(baseFeatures * 8 * 4 * 4, baseFeatures * 8),
            LeakyReLU.of(0.01),
            BatchNorm1d.of(baseFeatures * 8),
            Dropout.of(0.5)
        );

        fc6 = Sequential.of(
            Linear.of(baseFeatures * 8, baseFeatures * 4),
            LeakyReLU.of(0.01),
            BatchNorm1d.of(baseFeatures * 4),
            Dropout.of(0.5)
        );

        fc7 = Linear.of(baseFeatures * 4, numClasses);

        // Register modules
        network = Sequential.of(conv1, conv2, conv3, conv4, fc5, fc6, fc7);
    }

    private Sequential createConvBlock(int inChannels, int outChannels, boolean useStride) {
        int stride = useStride ? 2 : 1;

        return Sequential.of(
            // First conv with stride
            Conv3d.of(inChannels, outChannels, 3, stride, 1),
            BatchNorm3d.of(outChannels),
            LeakyReLU.of(0.01),

            // Second conv
            Conv3d.of(outChannels, outChannels, 3, 1, 1),
            LeakyReLU.of(0.01),

            // MaxPool
            MaxPool3d.of(2, 2)
        );
    }

    private Sequential createConvBlock4(int inChannels, int outChannels) {
        // Special block 4 with (1,3,3) kernel
        return Sequential.of(
            Conv3d.of(inChannels, outChannels, new int[]{1,3,3}, 1, new int[]{0,1,1}),
            BatchNorm3d.of(outChannels),
            LeakyReLU.of(0.01)
        );
    }

    @Override
    public Tensor forward(Tensor input) {
        // Conv blocks
        Tensor x = conv1.forward(input);
        x = conv2.forward(x);
        x = conv3.forward(x);
        x = conv4.forward(x);

        // Flatten
        x = x.view(-1, baseFeatures * 8 * 4 * 4);

        // Classifier
        x = fc5.forward(x);
        x = fc6.forward(x);
        x = fc7.forward(x);

        return x;
    }

    @Override
    public void initializeWeights() {
        // Kaiming initialization for Conv3d and Linear
        for (Parameter param : network.parameters()) {
            if (param.ndim() >= 2) {
                initKaimingNormal(param);
            }
        }
    }

    private void initKaimingNormal(Parameter param) {
        // PyTorch Kaiming normal initialization
        // Implementation using bytedeco bindings
    }
}
```

### 7.3 Generic 3D CNN

**File:** `vtea/deeplearning/models/Generic3DCNN.java`

**Implementation:**
```java
public class Generic3DCNN extends AbstractDeepLearningModel {

    private int[] convChannels;      // e.g., [32, 64, 128, 256]
    private int kernelSize;
    private String activationType;   // "ReLU", "LeakyReLU", "ELU"
    private String poolingType;      // "MaxPool3d", "AvgPool3d"
    private float dropout;
    private boolean useResidual;     // Use ResNet-style skip connections

    // Builder pattern
    public static class Builder {
        private int inputChannels = 1;
        private int numClasses = 2;
        private int[] convChannels = {32, 64, 128, 256};
        private int kernelSize = 3;
        private String activationType = "LeakyReLU";
        private String poolingType = "MaxPool3d";
        private float dropout = 0.5f;
        private boolean useResidual = false;

        public Builder inputChannels(int val) { inputChannels = val; return this; }
        public Builder numClasses(int val) { numClasses = val; return this; }
        public Builder convChannels(int[] val) { convChannels = val; return this; }
        public Builder kernelSize(int val) { kernelSize = val; return this; }
        public Builder activation(String val) { activationType = val; return this; }
        public Builder pooling(String val) { poolingType = val; return this; }
        public Builder dropout(float val) { dropout = val; return this; }
        public Builder useResidual(boolean val) { useResidual = val; return this; }

        public Generic3DCNN build() {
            return new Generic3DCNN(this);
        }
    }

    private Generic3DCNN(Builder builder) {
        this.modelName = "Generic3DCNN";
        this.inputChannels = builder.inputChannels;
        this.numClasses = builder.numClasses;
        this.convChannels = builder.convChannels;
        this.kernelSize = builder.kernelSize;
        this.activationType = builder.activationType;
        this.poolingType = builder.poolingType;
        this.dropout = builder.dropout;
        this.useResidual = builder.useResidual;

        buildArchitecture();
        initializeWeights();
    }

    @Override
    public void buildArchitecture() {
        // Build convolutional blocks
        ArrayList<Module> convBlocks = new ArrayList<>();
        int inChannels = inputChannels;

        for (int outChannels : convChannels) {
            if (useResidual && inChannels == outChannels) {
                convBlocks.add(createResidualBlock(inChannels, outChannels));
            } else {
                convBlocks.add(createConvBlock(inChannels, outChannels));
            }
            inChannels = outChannels;
        }

        // Build classifier
        int flattenedSize = convChannels[convChannels.length - 1] * 4 * 4 * 4; // Depends on input size
        Module classifier = createClassifier(flattenedSize);

        // Combine
        convBlocks.add(classifier);
        network = Sequential.of(convBlocks.toArray(new Module[0]));
    }

    private Module createConvBlock(int inChannels, int outChannels) {
        Module activation = createActivation();
        Module pooling = createPooling();

        return Sequential.of(
            Conv3d.of(inChannels, outChannels, kernelSize, 1, kernelSize/2),
            BatchNorm3d.of(outChannels),
            activation,
            Conv3d.of(outChannels, outChannels, kernelSize, 1, kernelSize/2),
            BatchNorm3d.of(outChannels),
            activation,
            pooling
        );
    }

    private Module createResidualBlock(int channels, int outChannels) {
        // ResNet-style skip connection
        return new ResidualBlock(channels, outChannels, kernelSize, createActivation());
    }

    private Module createClassifier(int inputSize) {
        int hidden = convChannels[convChannels.length - 1];

        return Sequential.of(
            Flatten.of(),
            Linear.of(inputSize, hidden),
            createActivation(),
            BatchNorm1d.of(hidden),
            Dropout.of(dropout),
            Linear.of(hidden, hidden / 2),
            createActivation(),
            BatchNorm1d.of(hidden / 2),
            Dropout.of(dropout),
            Linear.of(hidden / 2, numClasses)
        );
    }

    private Module createActivation() {
        switch (activationType) {
            case "ReLU": return ReLU.of();
            case "LeakyReLU": return LeakyReLU.of(0.01);
            case "ELU": return ELU.of();
            default: return LeakyReLU.of(0.01);
        }
    }

    private Module createPooling() {
        switch (poolingType) {
            case "MaxPool3d": return MaxPool3d.of(2, 2);
            case "AvgPool3d": return AvgPool3d.of(2, 2);
            default: return MaxPool3d.of(2, 2);
        }
    }

    // ... forward, initializeWeights implementations ...
}
```

**Usage Example:**
```java
Generic3DCNN model = new Generic3DCNN.Builder()
    .inputChannels(3)
    .numClasses(8)
    .convChannels(new int[]{32, 64, 128, 256})
    .kernelSize(3)
    .activation("LeakyReLU")
    .pooling("MaxPool3d")
    .dropout(0.5f)
    .useResidual(true)
    .build();
```

---

<a name="phase-5"></a>
## 8. Phase 5: Training Infrastructure

### 8.1 Trainer

**File:** `vtea/deeplearning/training/Trainer.java`

**Responsibilities:**
- Training loop with epochs and batching
- Loss computation and backpropagation
- Optimizer management
- Validation and metrics tracking
- Early stopping

**Implementation:**
```java
public class Trainer {

    private AbstractDeepLearningModel model;
    private DataLoader trainLoader;
    private DataLoader valLoader;
    private String lossFunction;      // "CrossEntropy", "FocalLoss"
    private String optimizer;         // "Adam", "SGD"
    private float learningRate;
    private int epochs;
    private boolean earlyStopping;
    private int patience;

    // Metrics tracking
    private ArrayList<Float> trainLosses = new ArrayList<>();
    private ArrayList<Float> valLosses = new ArrayList<>();
    private ArrayList<Float> valAccuracies = new ArrayList<>();

    public Trainer(AbstractDeepLearningModel model, TrainerConfig config) {
        this.model = model;
        this.lossFunction = config.lossFunction;
        this.optimizer = config.optimizer;
        this.learningRate = config.learningRate;
        this.epochs = config.epochs;
        this.earlyStopping = config.earlyStopping;
        this.patience = config.patience;
    }

    public void train(ArrayList<MicroObject> labeledObjects,
                      ImageStack[] imageStacks,
                      HashMap<MicroObject, Integer> labels) {

        // Create data loaders
        trainLoader = new DataLoader(labeledObjects, imageStacks, labels,
                                     config.batchSize, true);
        valLoader = createValidationLoader(labeledObjects, imageStacks, labels,
                                           config.valSplit);

        // Setup optimizer
        Optimizer opt = createOptimizer();

        // Setup loss function
        LossFunction loss = createLossFunction();

        int bestEpoch = 0;
        float bestValLoss = Float.MAX_VALUE;
        int patienceCounter = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Training phase
            model.train();
            float trainLoss = trainEpoch(trainLoader, opt, loss);
            trainLosses.add(trainLoss);

            // Validation phase
            model.eval();
            float valLoss = validateEpoch(valLoader, loss);
            float valAcc = computeAccuracy(valLoader);
            valLosses.add(valLoss);
            valAccuracies.add(valAcc);

            IJ.log(String.format("Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, Val Acc: %.2f%%",
                                epoch + 1, epochs, trainLoss, valLoss, valAcc * 100));

            // Early stopping check
            if (earlyStopping) {
                if (valLoss < bestValLoss) {
                    bestValLoss = valLoss;
                    bestEpoch = epoch;
                    patienceCounter = 0;
                    // Save best model
                    model.save("best_model_temp.pt");
                } else {
                    patienceCounter++;
                    if (patienceCounter >= patience) {
                        IJ.log("Early stopping triggered at epoch " + (epoch + 1));
                        model.load("best_model_temp.pt");
                        break;
                    }
                }
            }
        }

        IJ.log(String.format("Training complete. Best epoch: %d, Best val loss: %.4f",
                           bestEpoch + 1, bestValLoss));
    }

    private float trainEpoch(DataLoader loader, Optimizer opt, LossFunction loss) {
        float totalLoss = 0;
        int numBatches = 0;

        while (loader.hasNext()) {
            Batch batch = loader.next();

            // Forward pass
            Tensor predictions = model.forward(batch.inputs);
            Tensor lossValue = loss.compute(predictions, batch.labels);

            // Backward pass
            opt.zeroGrad();
            lossValue.backward();
            opt.step();

            totalLoss += lossValue.item();
            numBatches++;
        }

        return totalLoss / numBatches;
    }

    private float validateEpoch(DataLoader loader, LossFunction loss) {
        float totalLoss = 0;
        int numBatches = 0;

        try (NoGradGuard guard = new NoGradGuard()) {
            while (loader.hasNext()) {
                Batch batch = loader.next();
                Tensor predictions = model.forward(batch.inputs);
                Tensor lossValue = loss.compute(predictions, batch.labels);

                totalLoss += lossValue.item();
                numBatches++;
            }
        }

        return totalLoss / numBatches;
    }

    private float computeAccuracy(DataLoader loader) {
        int correct = 0;
        int total = 0;

        try (NoGradGuard guard = new NoGradGuard()) {
            while (loader.hasNext()) {
                Batch batch = loader.next();
                Tensor predictions = model.forward(batch.inputs);
                int[] predClasses = argmax(predictions);

                for (int i = 0; i < predClasses.length; i++) {
                    if (predClasses[i] == batch.labels.get(i)) {
                        correct++;
                    }
                    total++;
                }
            }
        }

        return (float) correct / total;
    }

    public float computeBalancedAccuracy(DataLoader loader, int numClasses) {
        int[] correctPerClass = new int[numClasses];
        int[] totalPerClass = new int[numClasses];

        try (NoGradGuard guard = new NoGradGuard()) {
            while (loader.hasNext()) {
                Batch batch = loader.next();
                Tensor predictions = model.forward(batch.inputs);
                int[] predClasses = argmax(predictions);

                for (int i = 0; i < predClasses.length; i++) {
                    int trueClass = batch.labels.get(i);
                    totalPerClass[trueClass]++;
                    if (predClasses[i] == trueClass) {
                        correctPerClass[trueClass]++;
                    }
                }
            }
        }

        // Compute per-class accuracy and average
        float sumAccuracy = 0;
        for (int i = 0; i < numClasses; i++) {
            if (totalPerClass[i] > 0) {
                sumAccuracy += (float) correctPerClass[i] / totalPerClass[i];
            }
        }

        return sumAccuracy / numClasses;
    }

    private Optimizer createOptimizer() {
        switch (optimizer) {
            case "Adam":
                return new Adam(model.parameters(), learningRate);
            case "SGD":
                return new SGD(model.parameters(), learningRate, 0.9); // momentum=0.9
            default:
                return new Adam(model.parameters(), learningRate);
        }
    }

    private LossFunction createLossFunction() {
        switch (lossFunction) {
            case "CrossEntropy":
                return new CrossEntropyLoss();
            case "FocalLoss":
                return new FocalLoss(2.0, null); // gamma=2.0
            default:
                return new CrossEntropyLoss();
        }
    }
}

class TrainerConfig {
    String lossFunction = "CrossEntropy";
    String optimizer = "Adam";
    float learningRate = 0.001f;
    int epochs = 100;
    int batchSize = 16;
    float valSplit = 0.2f;
    boolean earlyStopping = true;
    int patience = 10;
}
```

### 8.2 Data Loader

**File:** `vtea/deeplearning/training/DataLoader.java`

**Responsibilities:**
- Batch creation from MicroObjects
- Data augmentation
- Class balancing
- Multi-threading

**Implementation:**
```java
public class DataLoader implements Iterator<Batch> {

    private ArrayList<MicroObject> objects;
    private ImageStack[] imageStacks;
    private HashMap<MicroObject, Integer> labels;
    private int batchSize;
    private boolean shuffle;
    private boolean augment;
    private DatasetDefinition config;

    private int currentIndex = 0;
    private ArrayList<Integer> indices;

    public DataLoader(ArrayList<MicroObject> objects,
                      ImageStack[] imageStacks,
                      HashMap<MicroObject, Integer> labels,
                      int batchSize,
                      boolean shuffle) {
        this.objects = objects;
        this.imageStacks = imageStacks;
        this.labels = labels;
        this.batchSize = batchSize;
        this.shuffle = shuffle;

        // Initialize indices
        indices = new ArrayList<>();
        for (int i = 0; i < objects.size(); i++) {
            indices.add(i);
        }

        if (shuffle) {
            Collections.shuffle(indices);
        }
    }

    @Override
    public boolean hasNext() {
        return currentIndex < objects.size();
    }

    @Override
    public Batch next() {
        int endIndex = Math.min(currentIndex + batchSize, objects.size());
        List<Integer> batchIndices = indices.subList(currentIndex, endIndex);

        // Extract regions for this batch
        List<ImageStack[]> regions = new ArrayList<>();
        List<Integer> batchLabels = new ArrayList<>();

        for (int idx : batchIndices) {
            MicroObject obj = objects.get(idx);
            ImageStack[] region = CellRegionExtractor.extractRegion(
                obj, imageStacks, config.regionSize, config.channels
            );

            // Apply augmentation if enabled
            if (augment) {
                region = applyAugmentation(region);
            }

            regions.add(region);
            batchLabels.add(labels.get(obj));
        }

        // Convert to tensors
        Tensor inputTensor = TensorConverter.batchRegionsToTensor(regions);
        Tensor labelTensor = TensorConverter.listToTensor(batchLabels);

        currentIndex = endIndex;

        return new Batch(inputTensor, labelTensor);
    }

    private ImageStack[] applyAugmentation(ImageStack[] region) {
        Random rand = new Random();

        // Random rotation (90, 180, 270 degrees)
        if (rand.nextBoolean()) {
            region = rotate3D(region, rand.nextInt(4) * 90);
        }

        // Random flip
        if (rand.nextBoolean()) {
            region = flip3D(region, rand.nextInt(3)); // x, y, or z axis
        }

        // Random noise
        if (rand.nextFloat() < 0.3) {
            region = addGaussianNoise(region, 0.01f);
        }

        // Random brightness/contrast
        if (rand.nextFloat() < 0.3) {
            region = adjustBrightnessContrast(region,
                                              0.9f + rand.nextFloat() * 0.2f,
                                              0.9f + rand.nextFloat() * 0.2f);
        }

        return region;
    }

    public void reset() {
        currentIndex = 0;
        if (shuffle) {
            Collections.shuffle(indices);
        }
    }

    // Augmentation helper methods
    private ImageStack[] rotate3D(ImageStack[] stacks, int angle) { /* ... */ }
    private ImageStack[] flip3D(ImageStack[] stacks, int axis) { /* ... */ }
    private ImageStack[] addGaussianNoise(ImageStack[] stacks, float std) { /* ... */ }
    private ImageStack[] adjustBrightnessContrast(ImageStack[] stacks,
                                                   float brightness,
                                                   float contrast) { /* ... */ }
}

class Batch {
    Tensor inputs;
    Tensor labels;

    public Batch(Tensor inputs, Tensor labels) {
        this.inputs = inputs;
        this.labels = labels;
    }
}
```

### 8.3 Model Checkpoint (Updated with Class Definitions)

Already covered in Phase 3, Section 6.5.

---

<a name="phase-6"></a>
## 9. Phase 6: Integration with VTEA

### 9.1 Feature Processing Integration

**File:** `vtea/deeplearning/inference/DeepLearningClassification.java`

**Implementation:**
```java
@Plugin(type = FeatureProcessing.class)
public class DeepLearningClassification extends AbstractFeatureProcessing {

    private AbstractDeepLearningModel model;
    private DatasetDefinition datasetConfig;
    private String modelPath;

    public DeepLearningClassification() {
        VERSION = "1.0";
        AUTHOR = "VTEA Deep Learning Team";
        COMMENT = "3D Deep Learning Classification using PyTorch";
        NAME = "Deep Learning Classification";
        KEY = "DeepLearningClassification";
        TYPE = "Classifier";
    }

    public DeepLearningClassification(int max) {
        this();

        protocol = new ArrayList();

        // Model selection
        protocol.add(new JLabel("Model Path"));
        protocol.add(new JTextField("models/nephnet3d.pt", 30));
        protocol.add(new JButton("Browse..."));

        // Region size
        protocol.add(new JLabel("Region Size"));
        protocol.add(new JSpinner(new SpinnerNumberModel(64, 16, 256, 16)));

        // Channels
        protocol.add(new JLabel("Channels"));
        protocol.add(new JTextField("0,1,2", 10));

        // Batch size
        protocol.add(new JLabel("Batch Size"));
        protocol.add(new JSpinner(new SpinnerNumberModel(16, 1, 128, 1)));

        // Confidence threshold
        protocol.add(new JLabel("Confidence Threshold"));
        protocol.add(new JSpinner(new SpinnerNumberModel(0.5, 0.0, 1.0, 0.05)));
    }

    @Override
    public boolean process(ArrayList al, double[][] feature, boolean validate) {
        // Extract parameters
        modelPath = ((JTextField) al.get(1)).getText();
        int regionSize = ((Integer) ((JSpinner) al.get(4)).getValue());
        String channelsStr = ((JTextField) al.get(6)).getText();
        int batchSize = ((Integer) ((JSpinner) al.get(8)).getValue());
        float confThreshold = ((Double) ((JSpinner) al.get(10)).getValue()).floatValue();

        // Parse channels
        int[] channels = parseChannels(channelsStr);

        // Load model
        try {
            model = ModelCheckpoint.load(modelPath, "NephNet3D");
            IJ.log("Loaded model: " + model.getModelName());
            IJ.log("Classes: " + String.join(", ", model.getAllClassNames()));
        } catch (Exception e) {
            IJ.error("Failed to load model: " + e.getMessage());
            return false;
        }

        // Get objects and image data from VTEA context
        ArrayList<MicroObject> objects = getObjectsFromContext();
        ImageStack[] imageStacks = getImageStacksFromContext();

        // Extract regions and run inference
        ArrayList<String> classNames = new ArrayList<>();
        ArrayList<Float> confidences = new ArrayList<>();

        CellRegionExtractor extractor = new CellRegionExtractor();

        for (int i = 0; i < objects.size(); i++) {
            MicroObject obj = objects.get(i);

            // Extract region
            ImageStack[] region = extractor.extractRegion(
                obj, imageStacks,
                new int[]{regionSize, regionSize, regionSize},
                channels
            );

            // Convert to tensor
            Tensor input = TensorConverter.multiChannelToTensor(region, channels);

            // Run inference
            Tensor output = model.forward(input);
            Tensor probs = softmax(output);

            // Get prediction
            int predClass = argmax(output);
            float confidence = probs.get(predClass).item();

            // Store results
            if (confidence >= confThreshold) {
                classNames.add(model.getClassName(predClass));
                confidences.add(confidence);
            } else {
                classNames.add("Uncertain");
                confidences.add(confidence);
            }

            // Progress
            if (i % 100 == 0) {
                IJ.showProgress(i, objects.size());
                IJ.log(String.format("Processed %d/%d objects", i, objects.size()));
            }
        }

        // Store results
        dataResult.clear();
        dataResult.add(classNames);
        dataResult.add(confidences);

        IJ.showProgress(1.0);
        IJ.log("Classification complete!");

        return true;
    }

    @Override
    public String getDataDescription(ArrayList params) {
        String modelName = model != null ? model.getModelName() : "DL";
        return String.format("%s_Classification_%s", modelName, getCurrentTime());
    }

    private int[] parseChannels(String channelsStr) {
        String[] parts = channelsStr.split(",");
        int[] channels = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            channels[i] = Integer.parseInt(parts[i].trim());
        }
        return channels;
    }

    // Integration with VTEA context
    private ArrayList<MicroObject> getObjectsFromContext() {
        // Retrieve from VTEA session
        return VTEAContext.getInstance().getObjects();
    }

    private ImageStack[] getImageStacksFromContext() {
        // Retrieve from VTEA session
        return VTEAContext.getInstance().getImageStacks();
    }
}
```

### 9.2 Plugin Registration

**File:** `src/main/resources/META-INF/services/vtea.featureprocessing.FeatureProcessing`

Add line:
```
vtea.deeplearning.inference.DeepLearningClassification
```

### 9.3 Results Visualization

Integration with MicroExplorer (covered in Phase 3, Section 6.8).

---

<a name="phase-7"></a>
## 10. Phase 7: UI Components

### 10.1 Model Configuration Panel

**File:** `vtea/deeplearning/ui/ModelConfigPanel.java`

**Features:**
- Model type selection (NephNet3D, Generic3DCNN, Custom)
- Architecture parameter configuration
- Load/save model configurations
- Hyperparameter tuning interface

**Implementation:**
```java
public class ModelConfigPanel extends JPanel {

    private JComboBox<String> modelTypeCombo;
    private JSpinner inputChannelsSpinner;
    private JSpinner numClassesSpinner;
    private JTextField convChannelsField;
    private JComboBox<String> activationCombo;
    private JSpinner dropoutSpinner;

    public ModelConfigPanel() {
        setLayout(new GridBagLayout());
        initComponents();
    }

    private void initComponents() {
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(5, 5, 5, 5);

        // Model type
        gbc.gridx = 0; gbc.gridy = 0;
        add(new JLabel("Model Type:"), gbc);
        gbc.gridx = 1;
        modelTypeCombo = new JComboBox<>(new String[]{"NephNet3D", "Generic3DCNN"});
        modelTypeCombo.addActionListener(e -> updateConfigOptions());
        add(modelTypeCombo, gbc);

        // Input channels
        gbc.gridx = 0; gbc.gridy = 1;
        add(new JLabel("Input Channels:"), gbc);
        gbc.gridx = 1;
        inputChannelsSpinner = new JSpinner(new SpinnerNumberModel(1, 1, 10, 1));
        add(inputChannelsSpinner, gbc);

        // Number of classes
        gbc.gridx = 0; gbc.gridy = 2;
        add(new JLabel("Number of Classes:"), gbc);
        gbc.gridx = 1;
        numClassesSpinner = new JSpinner(new SpinnerNumberModel(2, 2, 100, 1));
        add(numClassesSpinner, gbc);

        // Conv channels (Generic3DCNN only)
        gbc.gridx = 0; gbc.gridy = 3;
        add(new JLabel("Conv Channels:"), gbc);
        gbc.gridx = 1;
        convChannelsField = new JTextField("32,64,128,256", 20);
        add(convChannelsField, gbc);

        // Activation function
        gbc.gridx = 0; gbc.gridy = 4;
        add(new JLabel("Activation:"), gbc);
        gbc.gridx = 1;
        activationCombo = new JComboBox<>(new String[]{"LeakyReLU", "ReLU", "ELU"});
        add(activationCombo, gbc);

        // Dropout
        gbc.gridx = 0; gbc.gridy = 5;
        add(new JLabel("Dropout:"), gbc);
        gbc.gridx = 1;
        dropoutSpinner = new JSpinner(new SpinnerNumberModel(0.5, 0.0, 0.9, 0.1));
        add(dropoutSpinner, gbc);

        // Buttons
        gbc.gridx = 0; gbc.gridy = 6; gbc.gridwidth = 2;
        JPanel buttonPanel = new JPanel();
        JButton loadButton = new JButton("Load Config");
        JButton saveButton = new JButton("Save Config");
        buttonPanel.add(loadButton);
        buttonPanel.add(saveButton);
        add(buttonPanel, gbc);
    }

    public AbstractDeepLearningModel createModel() {
        String modelType = (String) modelTypeCombo.getSelectedItem();
        int inputChannels = (Integer) inputChannelsSpinner.getValue();
        int numClasses = (Integer) numClassesSpinner.getValue();

        if ("NephNet3D".equals(modelType)) {
            return new NephNet3D(inputChannels, numClasses, 32);
        } else {
            int[] convChannels = parseConvChannels(convChannelsField.getText());
            String activation = (String) activationCombo.getSelectedItem();
            float dropout = ((Double) dropoutSpinner.getValue()).floatValue();

            return new Generic3DCNN.Builder()
                .inputChannels(inputChannels)
                .numClasses(numClasses)
                .convChannels(convChannels)
                .activation(activation)
                .dropout(dropout)
                .build();
        }
    }

    private int[] parseConvChannels(String text) {
        String[] parts = text.split(",");
        int[] channels = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            channels[i] = Integer.parseInt(parts[i].trim());
        }
        return channels;
    }
}
```

### 10.2 Training Interface

**File:** `vtea/deeplearning/ui/TrainingPanel.java`

**Features:**
- Load labeled data
- Training configuration (epochs, learning rate, batch size)
- Progress monitoring with live loss/accuracy plots
- Early stopping controls
- Model saving

**Implementation:**
```java
public class TrainingPanel extends JFrame {

    private ModelConfigPanel modelConfigPanel;
    private JTextField labeledDataPath;
    private JSpinner epochsSpinner;
    private JSpinner learningRateSpinner;
    private JSpinner batchSizeSpinner;
    private JCheckBox earlyStoppingCheckbox;
    private JButton trainButton;
    private JButton stopButton;

    private XYChart lossChart;
    private XYChart accuracyChart;
    private JProgressBar progressBar;
    private JTextArea logArea;

    private Trainer trainer;
    private volatile boolean training = false;

    public TrainingPanel() {
        super("Deep Learning Training");
        setLayout(new BorderLayout());
        initComponents();
        setSize(1200, 800);
        setLocationRelativeTo(null);
    }

    private void initComponents() {
        // Left panel: Configuration
        JPanel configPanel = new JPanel();
        configPanel.setLayout(new BoxLayout(configPanel, BoxLayout.Y_AXIS));
        configPanel.setBorder(BorderFactory.createTitledBorder("Configuration"));

        modelConfigPanel = new ModelConfigPanel();
        configPanel.add(modelConfigPanel);

        // Training parameters
        JPanel trainParamsPanel = new JPanel(new GridLayout(5, 2, 5, 5));
        trainParamsPanel.setBorder(BorderFactory.createTitledBorder("Training Parameters"));

        trainParamsPanel.add(new JLabel("Labeled Data:"));
        JPanel dataPanel = new JPanel(new BorderLayout());
        labeledDataPath = new JTextField();
        JButton browseButton = new JButton("Browse");
        browseButton.addActionListener(e -> browseLabeledData());
        dataPanel.add(labeledDataPath, BorderLayout.CENTER);
        dataPanel.add(browseButton, BorderLayout.EAST);
        trainParamsPanel.add(dataPanel);

        trainParamsPanel.add(new JLabel("Epochs:"));
        epochsSpinner = new JSpinner(new SpinnerNumberModel(100, 1, 1000, 10));
        trainParamsPanel.add(epochsSpinner);

        trainParamsPanel.add(new JLabel("Learning Rate:"));
        learningRateSpinner = new JSpinner(new SpinnerNumberModel(0.001, 0.0001, 0.1, 0.001));
        trainParamsPanel.add(learningRateSpinner);

        trainParamsPanel.add(new JLabel("Batch Size:"));
        batchSizeSpinner = new JSpinner(new SpinnerNumberModel(16, 1, 128, 1));
        trainParamsPanel.add(batchSizeSpinner);

        trainParamsPanel.add(new JLabel("Early Stopping:"));
        earlyStoppingCheckbox = new JCheckBox("Enabled", true);
        trainParamsPanel.add(earlyStoppingCheckbox);

        configPanel.add(trainParamsPanel);

        // Control buttons
        JPanel buttonPanel = new JPanel();
        trainButton = new JButton("Start Training");
        trainButton.addActionListener(e -> startTraining());
        stopButton = new JButton("Stop");
        stopButton.setEnabled(false);
        stopButton.addActionListener(e -> stopTraining());
        buttonPanel.add(trainButton);
        buttonPanel.add(stopButton);
        configPanel.add(buttonPanel);

        add(configPanel, BorderLayout.WEST);

        // Center panel: Charts
        JPanel chartsPanel = new JPanel(new GridLayout(2, 1));

        lossChart = createChart("Training Progress - Loss", "Epoch", "Loss");
        accuracyChart = createChart("Validation Accuracy", "Epoch", "Accuracy");

        chartsPanel.add(new XChartPanel<>(lossChart));
        chartsPanel.add(new XChartPanel<>(accuracyChart));

        add(chartsPanel, BorderLayout.CENTER);

        // Bottom panel: Progress and logs
        JPanel bottomPanel = new JPanel(new BorderLayout());

        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        bottomPanel.add(progressBar, BorderLayout.NORTH);

        logArea = new JTextArea(10, 80);
        logArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(logArea);
        bottomPanel.add(scrollPane, BorderLayout.CENTER);

        add(bottomPanel, BorderLayout.SOUTH);
    }

    private void startTraining() {
        training = true;
        trainButton.setEnabled(false);
        stopButton.setEnabled(true);

        // Create model
        AbstractDeepLearningModel model = modelConfigPanel.createModel();

        // Load labeled data
        ArrayList<MicroObject> objects = loadLabeledObjects();
        ImageStack[] imageStacks = loadImageStacks();
        HashMap<MicroObject, Integer> labels = loadLabels();

        // Create trainer config
        TrainerConfig config = new TrainerConfig();
        config.epochs = (Integer) epochsSpinner.getValue();
        config.learningRate = ((Double) learningRateSpinner.getValue()).floatValue();
        config.batchSize = (Integer) batchSizeSpinner.getValue();
        config.earlyStopping = earlyStoppingCheckbox.isSelected();

        // Create trainer
        trainer = new Trainer(model, config);

        // Run training in background thread
        new Thread(() -> {
            try {
                trainer.train(objects, imageStacks, labels);

                SwingUtilities.invokeLater(() -> {
                    JOptionPane.showMessageDialog(this, "Training complete!");
                    trainButton.setEnabled(true);
                    stopButton.setEnabled(false);

                    // Prompt to save model
                    int result = JOptionPane.showConfirmDialog(this,
                        "Save trained model?",
                        "Save Model",
                        JOptionPane.YES_NO_OPTION);
                    if (result == JOptionPane.YES_OPTION) {
                        saveModel(model);
                    }
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    JOptionPane.showMessageDialog(this,
                        "Training error: " + e.getMessage(),
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
                    trainButton.setEnabled(true);
                    stopButton.setEnabled(false);
                });
            }
        }).start();

        // Update charts periodically
        Timer chartUpdateTimer = new Timer(1000, e -> updateCharts());
        chartUpdateTimer.start();
    }

    private void stopTraining() {
        training = false;
        trainer.stop();
    }

    private void updateCharts() {
        if (trainer == null) return;

        // Update loss chart
        ArrayList<Float> trainLosses = trainer.getTrainLosses();
        ArrayList<Float> valLosses = trainer.getValLosses();

        lossChart.updateXYSeries("Train Loss",
                                 IntStream.range(0, trainLosses.size()).boxed().collect(Collectors.toList()),
                                 trainLosses, null);
        lossChart.updateXYSeries("Val Loss",
                                 IntStream.range(0, valLosses.size()).boxed().collect(Collectors.toList()),
                                 valLosses, null);

        // Update accuracy chart
        ArrayList<Float> valAccs = trainer.getValAccuracies();
        accuracyChart.updateXYSeries("Val Accuracy",
                                     IntStream.range(0, valAccs.size()).boxed().collect(Collectors.toList()),
                                     valAccs, null);

        // Update progress bar
        int currentEpoch = trainLosses.size();
        int totalEpochs = (Integer) epochsSpinner.getValue();
        progressBar.setValue((int) (100.0 * currentEpoch / totalEpochs));
        progressBar.setString(String.format("Epoch %d/%d", currentEpoch, totalEpochs));
    }

    private XYChart createChart(String title, String xLabel, String yLabel) {
        XYChart chart = new XYChartBuilder()
            .width(600)
            .height(300)
            .title(title)
            .xAxisTitle(xLabel)
            .yAxisTitle(yLabel)
            .build();

        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        chart.getStyler().setMarkerSize(5);

        return chart;
    }

    private void saveModel(AbstractDeepLearningModel model) {
        JFileChooser fc = new JFileChooser();
        fc.setDialogTitle("Save Model");
        fc.setFileFilter(new FileNameExtensionFilter("PyTorch Model (.pt)", "pt"));

        if (fc.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            String path = fc.getSelectedFile().getAbsolutePath();
            if (!path.endsWith(".pt")) {
                path = path.substring(0, path.lastIndexOf('.'));
            }

            ModelCheckpoint.save(model, path);
            logArea.append("Model saved to: " + path + "\n");
        }
    }
}
```

### 10.3 Inference Panel

Already covered in Phase 6, Section 9.1 (integrated into FeatureProcessing).

---

<a name="implementation-sequence"></a>
## 11. Implementation Sequence

### Sprint 1: Core Infrastructure (Week 1-2)
- [ ] Add bytedeco PyTorch dependencies to `pom.xml`
- [ ] Create package structure (`vtea/deeplearning/*`)
- [ ] Implement `TensorConverter.java`
- [ ] Unit tests for tensor conversions
- [ ] Verify PyTorch bindings work (simple tensor operations)

### Sprint 2: Data Pipeline (Week 3-4)
- [ ] Implement `CellRegionExtractor.java`
- [ ] Implement `DatasetDefinition.java`
- [ ] Unit tests for region extraction
- [ ] Test multi-channel extraction
- [ ] Performance benchmarking

### Sprint 3: Class Naming System (Week 5)
- [ ] Implement `ClassDefinition.java`
- [ ] Update `DatasetDefinition` with class mappings
- [ ] Enhance `ManualClassification.java` UI
- [ ] Implement JSON serialization for class definitions
- [ ] Test class definition import/export

### Sprint 4: NephNet3D Model (Week 6-7)
- [ ] Implement `AbstractDeepLearningModel.java`
- [ ] Implement `NephNet3D.java`
- [ ] Test forward pass with dummy data
- [ ] Verify architecture matches paper specs
- [ ] Test save/load functionality

### Sprint 5: Generic Architecture (Week 8)
- [ ] Implement `Generic3DCNN.java` with builder pattern
- [ ] Test various architecture configurations
- [ ] Compare performance with NephNet3D
- [ ] Documentation for architecture customization

### Sprint 6: Training Infrastructure (Week 9-11)
- [ ] Implement `Trainer.java`
- [ ] Implement `DataLoader.java` with augmentation
- [ ] Implement `ModelCheckpoint.java` (with class definitions)
- [ ] Test training on synthetic data
- [ ] Test early stopping and validation
- [ ] Performance optimization

### Sprint 7: VTEA Integration (Week 12-13)
- [ ] Implement `DeepLearningClassification.java` plugin
- [ ] Register in FeatureProcessing framework
- [ ] Integration with MicroExplorer visualization
- [ ] Test with real VTEA data
- [ ] End-to-end workflow testing

### Sprint 8: UI Components (Week 14-15)
- [ ] Implement `ModelConfigPanel.java`
- [ ] Implement `TrainingPanel.java` with live charts
- [ ] Integrate class naming UI
- [ ] User testing and feedback
- [ ] UI polish and refinements

### Sprint 9: Testing & Documentation (Week 16-17)
- [ ] Comprehensive unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] User documentation
- [ ] Developer documentation
- [ ] Example workflows

### Sprint 10: Validation & Release (Week 18)
- [ ] Validate with kidney tissue data
- [ ] Compare results with published NephNet3D
- [ ] Bug fixes and optimization
- [ ] Code review
- [ ] Release preparation

---

<a name="challenges"></a>
## 12. Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Memory constraints** for 3D data | Implement lazy loading, batch processing, crop smaller regions (32³ or 64³ instead of 128³) |
| **Bytedeco API learning curve** | Reference bytedeco examples, PyTorch C++ API docs, create utility wrapper classes |
| **Class imbalance** in kidney data | Weighted loss functions, focal loss, data augmentation, balanced sampling |
| **Long training times** | GPU acceleration, mixed precision training, start with smaller models, transfer learning |
| **Model portability** | ONNX export, save architecture JSON + weights separately, version control |
| **Integration complexity** | Maintain backward compatibility, optional feature flag, thorough testing |
| **3D augmentation challenges** | Use proven augmentation libraries, limit to safe transformations (rotation, flip) |
| **Hyperparameter tuning** | Implement grid search, random search, or Bayesian optimization |
| **Large model file sizes** | Model compression, quantization, separate model repository |
| **Cross-platform compatibility** | Test on Windows, Linux, macOS; bytedeco provides platform-specific binaries |

---

<a name="questions"></a>
## 13. Missing Components / Questions

1. **GPU Support**:
   - Should we require GPU or support CPU-only mode?
   - How to handle systems without CUDA?
   - **Recommendation**: Support both, auto-detect GPU, graceful fallback to CPU

2. **Pretrained Models**:
   - Should we include pretrained NephNet3D weights from the paper?
   - How to distribute large model files?
   - **Recommendation**: Separate model repository, download on-demand

3. **Data Labeling**:
   - Use existing `ManualClassification` or create new labeling tool?
   - **Decision**: Enhance existing tool with class naming (Phase 3)

4. **Multi-scale Processing**:
   - Should regions be extracted at multiple scales?
   - **Recommendation**: Future enhancement, start with single scale

5. **Ensemble Methods**:
   - Support for model ensembles?
   - **Recommendation**: Future enhancement after basic implementation

6. **Active Learning**:
   - Priority labeling for uncertain predictions?
   - **Recommendation**: Future enhancement

7. **Model Zoo**:
   - Infrastructure for sharing/loading community models?
   - **Recommendation**: Phase 2 feature, implement after core functionality

8. **Validation Strategy**:
   - K-fold cross-validation support?
   - **Recommendation**: Implement in Trainer as optional feature

9. **Class Naming Migration**:
   - How to handle existing clustering results without class names?
   - **Decision**: Backward compatibility with auto-generated names (Phase 3, Section 6.10)

10. **Multi-user Collaboration**:
    - Share class definitions across team?
    - **Recommendation**: JSON export/import enables sharing

---

<a name="testing"></a>
## 14. Testing Strategy

### 14.1 Unit Tests

**Coverage:**
- `TensorConverter`: All conversion methods, normalization, edge cases
- `CellRegionExtractor`: Boundary conditions, padding, multi-channel
- `ClassDefinition`: Serialization, validation, color generation
- `NephNet3D`: Forward pass, architecture validation
- `Generic3DCNN`: Builder pattern, various configurations
- `DataLoader`: Batching, shuffling, augmentation

**Tools:** JUnit 5, Mockito for mocking ImageJ components

### 14.2 Integration Tests

**Scenarios:**
1. **End-to-End Training**:
   - Load labeled data → Train model → Save checkpoint → Load checkpoint → Inference
2. **VTEA Plugin Integration**:
   - Register plugin → Load in VTEA → Process objects → Visualize results
3. **Class Naming Workflow**:
   - Define classes → Label data → Train → Inference → Verify names in results
4. **Multi-channel Processing**:
   - Extract regions from 3-channel stack → Train → Inference

### 14.3 Performance Tests

**Benchmarks:**
- **Tensor Conversion**: Time to convert 1000 regions to tensors
- **Inference Speed**: Objects/second for batch sizes 1, 16, 32, 64
- **Memory Usage**: Peak memory for training on 1000 samples
- **GPU vs CPU**: Speedup factor

**Targets:**
- Inference: >10 objects/second on CPU, >100 objects/second on GPU
- Memory: <8GB for training batch size 16
- Tensor conversion: <1ms per region

### 14.4 Validation Tests

**Scientific Validation:**
- **NephNet3D Replication**: Compare accuracy on test set with published results
  - Target: >75% balanced accuracy (paper reports 80.26%)
- **Class Distribution**: Verify predictions match known cell type distributions
- **Visual Inspection**: Manual review of classified cells

### 14.5 User Acceptance Testing

**Test Cases:**
1. Researcher labels 100 cells with custom names → Trains model → Achieves >70% accuracy
2. Load pretrained model → Classify 1000 cells → Export results CSV
3. Modify Generic3DCNN architecture → Train → Compare performance
4. Import class definitions from colleague → Use for labeling

---

<a name="documentation"></a>
## 15. Documentation Needs

### 15.1 User Documentation

**User Guide:**
1. **Installation**
   - Add VTEA plugin
   - Install dependencies
   - GPU setup (optional)

2. **Quick Start**
   - Load image
   - Segment cells
   - Define class names
   - Label training data
   - Train model
   - Run classification

3. **Tutorials**
   - Kidney tissue classification (NephNet3D)
   - Custom tissue type (Generic3DCNN)
   - Class definition best practices
   - Model sharing workflow

4. **Troubleshooting**
   - Common errors
   - Performance optimization
   - Memory issues

### 15.2 Developer Documentation

**Developer Guide:**
1. **Architecture Overview**
   - Package structure
   - Class hierarchy diagrams
   - Data flow diagrams

2. **Extending the Framework**
   - Create custom architecture
   - Add new loss functions
   - Custom data augmentation

3. **API Reference**
   - Javadoc for all public classes
   - Method signatures and examples

4. **ByteDeco PyTorch Guide**
   - Common operations
   - Tensor manipulation
   - Model building patterns

### 15.3 Example Notebooks

**Jupyter Notebooks (Optional):**
1. Basic tensor operations
2. Region extraction visualization
3. Model architecture comparison
4. Training curves analysis
5. Prediction visualization

### 15.4 Video Tutorials

**Screencasts:**
1. Installation and setup (5 min)
2. Training your first model (10 min)
3. Advanced: Custom architectures (15 min)
4. Class naming and labeling (8 min)

---

## Appendices

### Appendix A: File Structure

```
volumetric-tissue-exploration-analysis/
├── pom.xml (updated with bytedeco dependencies)
├── src/main/java/
│   └── vtea/
│       ├── deeplearning/
│       │   ├── models/
│       │   │   ├── AbstractDeepLearningModel.java
│       │   │   ├── NephNet3D.java
│       │   │   └── Generic3DCNN.java
│       │   ├── data/
│       │   │   ├── TensorConverter.java
│       │   │   ├── DatasetDefinition.java
│       │   │   ├── ClassDefinition.java
│       │   │   └── CellRegionExtractor.java
│       │   ├── training/
│       │   │   ├── Trainer.java
│       │   │   ├── DataLoader.java
│       │   │   └── ModelCheckpoint.java
│       │   ├── inference/
│       │   │   └── DeepLearningClassification.java
│       │   └── ui/
│       │       ├── ModelConfigPanel.java
│       │       ├── TrainingPanel.java
│       │       └── ClassDefinitionPanel.java
│       └── exploration/
│           └── plottools/
│               └── panels/
│                   └── ManualClassification.java (enhanced)
├── src/test/java/
│   └── vtea/deeplearning/
│       ├── TensorConverterTest.java
│       ├── CellRegionExtractorTest.java
│       ├── NephNet3DTest.java
│       └── IntegrationTest.java
└── docs/
    ├── USER_GUIDE.md
    ├── DEVELOPER_GUIDE.md
    ├── API_REFERENCE.md
    └── tutorials/
        ├── quickstart.md
        ├── kidney_classification.md
        └── custom_architecture.md
```

### Appendix B: Dependencies

**Required:**
- bytedeco/pytorch-platform: 2.9.1-1.5.11
- gson: 2.10.1
- ImageJ: 1.53s (already included)
- imglib2: 5.12.0 (already included)

**Optional:**
- bytedeco/cuda-platform: 12.3-8.9-1.5.11 (GPU support)
- xchart: 3.8.1 (for training charts, already included)

### Appendix C: References

1. **NephNet3D Paper**:
   - Woloshuk A, et al. "In Situ Classification of Cell Types in Human Kidney Tissue Using 3D Nuclear Staining." *Cytometry Part A*, 2020.
   - [PubMed](https://pubmed.ncbi.nlm.nih.gov/33252180/)
   - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8382162/)

2. **Bytedeco PyTorch**:
   - [GitHub Repository](https://github.com/bytedeco/javacpp-presets/tree/master/pytorch)
   - [Documentation](https://github.com/bytedeco/javacpp-presets/wiki/PyTorch)

3. **VTEA**:
   - [GitHub Repository](https://github.com/icbm-iupui/volumetric-tissue-exploration-analysis)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-26
**Status:** Ready for Implementation
