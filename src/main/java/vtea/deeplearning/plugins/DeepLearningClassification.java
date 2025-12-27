/*
 * Copyright (C) 2025 University of Nebraska
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package vtea.deeplearning.plugins;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import org.bytedeco.pytorch.Tensor;
import org.scijava.plugin.Plugin;
import vtea.deeplearning.data.CellRegionExtractor;
import vtea.deeplearning.data.DatasetDefinition;
import vtea.deeplearning.data.TensorConverter;
import vtea.deeplearning.models.AbstractDeepLearningModel;
import vtea.deeplearning.models.NephNet3D;
import vtea.deeplearning.models.Generic3DCNN;
import vtea.featureprocessing.AbstractFeatureProcessing;
import vtea.featureprocessing.FeatureProcessing;
import vtea.objects.layercake.microObject;
import vtea.protocol.MicroBlockStep;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Deep Learning Classification plugin for VTEA.
 * Classifies MicroObjects using trained 3D CNN models.
 *
 * Features:
 * - Load trained models (NephNet3D, Generic3DCNN)
 * - Extract 3D regions around segmented cells
 * - Batch inference for efficiency
 * - Return class IDs or human-readable class names
 * - Progress tracking
 *
 * @author VTEA Deep Learning Team
 */
@Plugin(type = FeatureProcessing.class)
public class DeepLearningClassification extends AbstractFeatureProcessing<Component, Object> {

    private AbstractDeepLearningModel model;
    private DatasetDefinition datasetDef;
    private List<microObject> objects;
    private ImageStack[] imageStacks;

    /**
     * Creates the Comment Text for the Block GUI.
     */
    public static String getBlockComment(ArrayList comComponents) {
        String comment = "<html>";
        comment = comment.concat("Deep Learning Classification<br>");
        comment = comment.concat("Model: " + getModelName(comComponents) + "<br>");
        comment = comment.concat("</html>");
        return comment;
    }

    /**
     * Get model name from components
     */
    private static String getModelName(ArrayList comComponents) {
        if (comComponents.size() > 1 && comComponents.get(1) instanceof JTextField) {
            String path = ((JTextField) comComponents.get(1)).getText();
            if (path != null && !path.isEmpty()) {
                return new File(path).getName();
            }
        }
        return "Not selected";
    }

    /**
     * Basic Constructor
     */
    public DeepLearningClassification() {
        VERSION = "1.0";
        AUTHOR = "VTEA Deep Learning Team";
        COMMENT = "3D deep learning classification using PyTorch models";
        NAME = "Deep Learning Classification";
        KEY = "DeepLearningClassification";
        TYPE = "Cluster";  // Classification acts like clustering in VTEA
    }

    /**
     * Constructor with max objects
     */
    public DeepLearningClassification(int max) {
        this();
        dataResult = new ArrayList(max);
    }

    @Override
    public String getDataDescription(ArrayList params) {
        String modelName = getModelName(params);
        return KEY + '_' + modelName + '_' + getCurrentTime();
    }

    /**
     * Main processing method
     *
     * @param al Parameters from UI
     * @param feature Feature array [objects x features]
     * @param validate Whether to perform validation
     * @return true when complete
     */
    @Override
    public boolean process(ArrayList al, double[][] feature, boolean validate) {
        try {
            long startTime = System.nanoTime();
            progress = 0;

            IJ.log("=".repeat(80));
            IJ.log("Deep Learning Classification");
            IJ.log("=".repeat(80));

            // Extract parameters
            String modelPath = extractModelPath(al);
            boolean returnClassNames = extractReturnClassNames(al);
            int batchSize = extractBatchSize(al);

            // Get image data from protocol
            if (!extractImageData(al)) {
                IJ.log("ERROR: Could not extract image data from protocol");
                return false;
            }

            // Load model
            IJ.log("Loading model from: " + modelPath);
            if (!loadModel(modelPath)) {
                IJ.log("ERROR: Failed to load model");
                return false;
            }

            IJ.log("Model: " + model.getModelName());
            IJ.log("Input channels: " + model.getInputChannels());
            IJ.log("Number of classes: " + model.getNumClasses());
            IJ.log("Class names: " + model.getClassDefinitions().keySet());

            // Load dataset definition (stored with model)
            if (!loadDatasetDefinition(modelPath)) {
                IJ.log("WARNING: Could not load dataset definition, using defaults");
                createDefaultDatasetDefinition();
            }

            IJ.log("Dataset region size: " + java.util.Arrays.toString(datasetDef.getRegionSize()));
            IJ.log("Channels: " + java.util.Arrays.toString(datasetDef.getChannels()));
            IJ.log("Normalization: " + datasetDef.getNormalizationType());

            // Get objects from feature array
            extractObjects(feature);
            IJ.log("Processing " + objects.size() + " objects");

            // Run inference
            IJ.log("Running inference...");
            List<Integer> classifications = classifyObjects(batchSize);

            // Store results
            dataResult.clear();
            dataResult.ensureCapacity(classifications.size());

            if (returnClassNames) {
                // Return class names
                for (int classId : classifications) {
                    String className = model.getClassName(classId);
                    dataResult.add(className);
                }
                IJ.log("Returning class names");
            } else {
                // Return class IDs
                for (int classId : classifications) {
                    dataResult.add((double) classId);
                }
                IJ.log("Returning class IDs");
            }

            // Print class distribution
            printClassDistribution(classifications);

            long endTime = System.nanoTime();
            IJ.log("PROFILING: Classification completed in " + (endTime - startTime) / 1000000 + " ms");
            IJ.log("=".repeat(80));

            progress = 100;
            return true;

        } catch (Exception e) {
            IJ.log("ERROR: Deep learning classification failed: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Extract model path from parameters
     */
    private String extractModelPath(ArrayList al) {
        if (al.size() > 1 && al.get(1) instanceof JTextField) {
            return ((JTextField) al.get(1)).getText();
        }
        return "";
    }

    /**
     * Extract whether to return class names
     */
    private boolean extractReturnClassNames(ArrayList al) {
        if (al.size() > 2 && al.get(2) instanceof JCheckBox) {
            return ((JCheckBox) al.get(2)).isSelected();
        }
        return true;  // Default to class names
    }

    /**
     * Extract batch size
     */
    private int extractBatchSize(ArrayList al) {
        if (al.size() > 3 && al.get(3) instanceof JSpinner) {
            return ((Integer) ((JSpinner) al.get(3)).getValue());
        }
        return 16;  // Default batch size
    }

    /**
     * Extract image data from protocol
     */
    private boolean extractImageData(ArrayList al) {
        // Look for MicroBlockStep in parameters
        for (Object param : al) {
            if (param instanceof MicroBlockStep) {
                MicroBlockStep step = (MicroBlockStep) param;

                // Get image stacks from the step
                ImagePlus[] images = step.getImagePluses();
                if (images != null && images.length > 0) {
                    imageStacks = new ImageStack[images.length];
                    for (int i = 0; i < images.length; i++) {
                        imageStacks[i] = images[i].getImageStack();
                    }
                    return true;
                }
            }
        }

        // Try to get from current ImagePlus
        ImagePlus imp = IJ.getImage();
        if (imp != null) {
            imageStacks = new ImageStack[]{imp.getImageStack()};
            return true;
        }

        return false;
    }

    /**
     * Load trained model
     */
    private boolean loadModel(String modelPath) {
        try {
            // Try to determine model type from metadata
            String metaPath = modelPath + ".meta";
            File metaFile = new File(metaPath);

            if (metaFile.exists()) {
                // Read metadata to get model class
                // For now, try NephNet3D first, then Generic3DCNN
                try {
                    model = new NephNet3D(1, 2);  // Dummy params, will be overwritten
                    model.load(modelPath);
                    return true;
                } catch (Exception e1) {
                    try {
                        model = Generic3DCNN.createLightweight(1, 2, new int[]{64, 64, 64});
                        model.load(modelPath);
                        return true;
                    } catch (Exception e2) {
                        IJ.log("ERROR: Could not load model as NephNet3D or Generic3DCNN");
                        e2.printStackTrace();
                        return false;
                    }
                }
            } else {
                IJ.log("ERROR: Model metadata file not found: " + metaPath);
                return false;
            }

        } catch (Exception e) {
            IJ.log("ERROR: Failed to load model: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Load dataset definition
     */
    private boolean loadDatasetDefinition(String modelPath) {
        try {
            String datasetPath = modelPath + "_dataset.json";
            File datasetFile = new File(datasetPath);

            if (datasetFile.exists()) {
                datasetDef = DatasetDefinition.loadFromFile(datasetPath);
                return true;
            }
            return false;

        } catch (Exception e) {
            IJ.log("WARNING: Could not load dataset definition: " + e.getMessage());
            return false;
        }
    }

    /**
     * Create default dataset definition
     */
    private void createDefaultDatasetDefinition() {
        int[] inputShape = model.getInputShape();
        int inputChannels = model.getInputChannels();

        int[] channels = new int[inputChannels];
        for (int i = 0; i < inputChannels; i++) {
            channels[i] = i;
        }

        datasetDef = new DatasetDefinition.Builder()
            .name("Default Dataset")
            .regionSize(inputShape[0], inputShape[1], inputShape[2])
            .channels(channels)
            .normalize(true)
            .normalizationType(TensorConverter.NormalizationType.ZSCORE)
            .paddingType(CellRegionExtractor.PaddingType.ZERO)
            .build();

        datasetDef.setClassDefinitions(model.getClassDefinitions());
    }

    /**
     * Extract MicroObjects from feature array
     */
    private void extractObjects(double[][] feature) {
        objects = new ArrayList<>();

        // Feature array first column contains object IDs
        // We need to get the actual MicroObject instances
        // For now, create placeholder objects with centroids from features
        // In real integration, these would come from the protocol

        for (int i = 0; i < feature.length; i++) {
            // Assuming features include X, Y, Z coordinates
            // This is a simplified version - real implementation would get actual MicroObjects
            microObject obj = new microObject();

            // Try to extract coordinates from features
            // Typical feature array: [ID, X, Y, Z, ...]
            if (feature[i].length >= 4) {
                obj.setX((int) feature[i][1]);
                obj.setY((int) feature[i][2]);
                obj.setZ((int) feature[i][3]);
            }

            objects.add(obj);
        }
    }

    /**
     * Classify all objects
     */
    private List<Integer> classifyObjects(int batchSize) {
        List<Integer> classifications = new ArrayList<>();
        int totalObjects = objects.size();

        // Process in batches
        for (int batchStart = 0; batchStart < totalObjects; batchStart += batchSize) {
            int batchEnd = Math.min(batchStart + batchSize, totalObjects);
            List<microObject> batchObjects = objects.subList(batchStart, batchEnd);

            // Extract regions for batch
            List<ImageStack[]> batchRegions = new ArrayList<>();
            for (microObject obj : batchObjects) {
                try {
                    ImageStack[] region = CellRegionExtractor.extractRegion(
                        obj,
                        imageStacks,
                        datasetDef.getRegionSize(),
                        datasetDef.getChannels(),
                        datasetDef.getPaddingType()
                    );
                    batchRegions.add(region);
                } catch (Exception e) {
                    IJ.log("WARNING: Could not extract region for object at (" +
                           obj.getX() + ", " + obj.getY() + ", " + obj.getZ() + ")");
                    // Add null placeholder
                    batchRegions.add(null);
                }
            }

            // Convert to tensor and predict
            if (!batchRegions.isEmpty()) {
                // Filter out nulls
                batchRegions.removeIf(r -> r == null);

                if (!batchRegions.isEmpty()) {
                    Tensor batchTensor = TensorConverter.batchRegionsToTensor(
                        batchRegions,
                        datasetDef.getChannels(),
                        datasetDef.getNormalizationType()
                    );

                    // Predict
                    int[] predictions = model.predict(batchTensor);

                    // Add to results
                    for (int pred : predictions) {
                        classifications.add(pred);
                    }

                    batchTensor.close();
                }
            }

            // Update progress
            progress = (int) (100.0 * batchEnd / totalObjects);
            IJ.showProgress(batchEnd, totalObjects);
        }

        return classifications;
    }

    /**
     * Print class distribution
     */
    private void printClassDistribution(List<Integer> classifications) {
        java.util.Map<Integer, Integer> distribution = new java.util.HashMap<>();

        for (int classId : classifications) {
            distribution.put(classId, distribution.getOrDefault(classId, 0) + 1);
        }

        IJ.log("\nClass Distribution:");
        for (int classId : distribution.keySet()) {
            String className = model.getClassName(classId);
            int count = distribution.get(classId);
            double percentage = 100.0 * count / classifications.size();
            IJ.log(String.format("  %s (ID=%d): %d objects (%.1f%%)",
                                className, classId, count, percentage));
        }
    }

    @Override
    public boolean copyComponentParameter(int index, ArrayList dComponents, ArrayList sComponents) {
        try {
            if (index == 1 && dComponents.get(1) instanceof JTextField && sComponents.get(1) instanceof JTextField) {
                // Copy model path
                ((JTextField) dComponents.get(1)).setText(((JTextField) sComponents.get(1)).getText());
            }
            if (index == 2 && dComponents.get(2) instanceof JCheckBox && sComponents.get(2) instanceof JCheckBox) {
                // Copy return class names checkbox
                ((JCheckBox) dComponents.get(2)).setSelected(((JCheckBox) sComponents.get(2)).isSelected());
            }
            if (index == 3 && dComponents.get(3) instanceof JSpinner && sComponents.get(3) instanceof JSpinner) {
                // Copy batch size
                ((JSpinner) dComponents.get(3)).setValue(((JSpinner) sComponents.get(3)).getValue());
            }
            return true;
        } catch (Exception e) {
            IJ.log("ERROR: Could not copy parameter for Deep Learning Classification");
            return false;
        }
    }
}
