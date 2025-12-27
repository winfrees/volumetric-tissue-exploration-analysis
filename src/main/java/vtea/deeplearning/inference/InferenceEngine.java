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
package vtea.deeplearning.inference;

import ij.ImageStack;
import org.bytedeco.pytorch.Tensor;
import vtea.deeplearning.data.CellRegionExtractor;
import vtea.deeplearning.data.DatasetDefinition;
import vtea.deeplearning.data.TensorConverter;
import vtea.deeplearning.models.AbstractDeepLearningModel;
import vtea.objects.layercake.microObject;

import java.util.*;

/**
 * Inference engine for running deep learning predictions on MicroObjects.
 * Provides high-level interface for classification tasks.
 *
 * Features:
 * - Batch processing for efficiency
 * - Progress callbacks
 * - Error handling for edge cases
 * - Memory-efficient processing
 *
 * @author VTEA Deep Learning Team
 */
public class InferenceEngine {

    private final AbstractDeepLearningModel model;
    private final DatasetDefinition datasetDef;
    private final int batchSize;
    private final boolean verbose;

    // Progress callback
    private ProgressCallback progressCallback;

    /**
     * Interface for progress callbacks
     */
    public interface ProgressCallback {
        void onProgress(int current, int total, String message);
    }

    /**
     * Result container for inference
     */
    public static class InferenceResult {
        public final List<Integer> classIds;
        public final List<String> classNames;
        public final List<float[]> probabilities;
        public final List<microObject> objects;
        public final Map<Integer, Integer> classDistribution;

        public InferenceResult(List<Integer> classIds, List<String> classNames,
                             List<float[]> probabilities, List<microObject> objects) {
            this.classIds = classIds;
            this.classNames = classNames;
            this.probabilities = probabilities;
            this.objects = objects;
            this.classDistribution = calculateDistribution(classIds);
        }

        private Map<Integer, Integer> calculateDistribution(List<Integer> classIds) {
            Map<Integer, Integer> dist = new HashMap<>();
            for (int classId : classIds) {
                dist.put(classId, dist.getOrDefault(classId, 0) + 1);
            }
            return dist;
        }

        public int getTotalObjects() {
            return classIds.size();
        }

        public double getClassPercentage(int classId) {
            int count = classDistribution.getOrDefault(classId, 0);
            return 100.0 * count / getTotalObjects();
        }
    }

    /**
     * Constructor
     */
    public InferenceEngine(AbstractDeepLearningModel model, DatasetDefinition datasetDef,
                          int batchSize, boolean verbose) {
        this.model = model;
        this.datasetDef = datasetDef;
        this.batchSize = batchSize;
        this.verbose = verbose;

        if (!model.isBuilt()) {
            model.build();
        }
    }

    /**
     * Constructor with default batch size
     */
    public InferenceEngine(AbstractDeepLearningModel model, DatasetDefinition datasetDef) {
        this(model, datasetDef, 16, false);
    }

    /**
     * Set progress callback
     */
    public void setProgressCallback(ProgressCallback callback) {
        this.progressCallback = callback;
    }

    /**
     * Run inference on a list of MicroObjects
     *
     * @param objects List of MicroObjects to classify
     * @param imageStacks Image stacks containing the cell data
     * @return InferenceResult containing classifications and probabilities
     */
    public InferenceResult predict(List<microObject> objects, ImageStack[] imageStacks) {
        if (verbose) {
            System.out.println("Starting inference on " + objects.size() + " objects");
        }

        model.eval();

        List<Integer> allClassIds = new ArrayList<>();
        List<String> allClassNames = new ArrayList<>();
        List<float[]> allProbabilities = new ArrayList<>();
        List<microObject> validObjects = new ArrayList<>();

        int totalObjects = objects.size();
        int processedObjects = 0;

        // Process in batches
        for (int batchStart = 0; batchStart < totalObjects; batchStart += batchSize) {
            int batchEnd = Math.min(batchStart + batchSize, totalObjects);
            List<microObject> batchObjects = objects.subList(batchStart, batchEnd);

            // Extract regions for this batch
            List<ImageStack[]> batchRegions = new ArrayList<>();
            List<microObject> batchValidObjects = new ArrayList<>();

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
                    batchValidObjects.add(obj);

                } catch (Exception e) {
                    if (verbose) {
                        System.err.println("WARNING: Could not extract region for object at (" +
                                         obj.getX() + ", " + obj.getY() + ", " + obj.getZ() + "): " +
                                         e.getMessage());
                    }
                    // Skip this object
                }
            }

            // Run inference on batch
            if (!batchRegions.isEmpty()) {
                Tensor batchTensor = TensorConverter.batchRegionsToTensor(
                    batchRegions,
                    datasetDef.getChannels(),
                    datasetDef.getNormalizationType()
                );

                // Get predictions and probabilities
                int[] classIds = model.predict(batchTensor);
                float[][] probabilities = model.predictProba(batchTensor);

                // Store results
                for (int i = 0; i < classIds.length; i++) {
                    int classId = classIds[i];
                    String className = model.getClassName(classId);

                    allClassIds.add(classId);
                    allClassNames.add(className);
                    allProbabilities.add(probabilities[i]);
                    validObjects.add(batchValidObjects.get(i));
                }

                batchTensor.close();
            }

            processedObjects = batchEnd;

            // Progress callback
            if (progressCallback != null) {
                String message = String.format("Processed %d/%d objects", processedObjects, totalObjects);
                progressCallback.onProgress(processedObjects, totalObjects, message);
            }
        }

        if (verbose) {
            System.out.println("Inference complete: " + validObjects.size() + " objects classified");
        }

        return new InferenceResult(allClassIds, allClassNames, allProbabilities, validObjects);
    }

    /**
     * Run inference and return only class IDs
     */
    public List<Integer> predictClassIds(List<microObject> objects, ImageStack[] imageStacks) {
        return predict(objects, imageStacks).classIds;
    }

    /**
     * Run inference and return only class names
     */
    public List<String> predictClassNames(List<microObject> objects, ImageStack[] imageStacks) {
        return predict(objects, imageStacks).classNames;
    }

    /**
     * Run inference on a single object
     */
    public int predictSingle(microObject object, ImageStack[] imageStacks) {
        List<microObject> singleObjectList = Collections.singletonList(object);
        InferenceResult result = predict(singleObjectList, imageStacks);
        return result.classIds.isEmpty() ? -1 : result.classIds.get(0);
    }

    /**
     * Get prediction with probabilities for a single object
     */
    public AbstractDeepLearningModel.PredictionResult predictSingleDetailed(microObject object,
                                                                            ImageStack[] imageStacks) {
        try {
            ImageStack[] region = CellRegionExtractor.extractRegion(
                object,
                imageStacks,
                datasetDef.getRegionSize(),
                datasetDef.getChannels(),
                datasetDef.getPaddingType()
            );

            List<ImageStack[]> singleRegion = Collections.singletonList(region);

            Tensor tensor = TensorConverter.batchRegionsToTensor(
                singleRegion,
                datasetDef.getChannels(),
                datasetDef.getNormalizationType()
            );

            AbstractDeepLearningModel.PredictionResult[] results = model.predictDetailed(tensor);
            tensor.close();

            return results.length > 0 ? results[0] : null;

        } catch (Exception e) {
            if (verbose) {
                System.err.println("ERROR: Prediction failed for object: " + e.getMessage());
            }
            return null;
        }
    }

    /**
     * Batch predict with detailed results
     */
    public List<AbstractDeepLearningModel.PredictionResult> predictDetailed(List<microObject> objects,
                                                                            ImageStack[] imageStacks) {
        InferenceResult result = predict(objects, imageStacks);
        List<AbstractDeepLearningModel.PredictionResult> detailedResults = new ArrayList<>();

        for (int i = 0; i < result.classIds.size(); i++) {
            int classId = result.classIds.get(i);
            String className = result.classNames.get(i);
            float[] probs = result.probabilities.get(i);
            float probability = probs[classId];

            AbstractDeepLearningModel.PredictionResult pred =
                new AbstractDeepLearningModel.PredictionResult(classId, className, probability, probs);

            detailedResults.add(pred);
        }

        return detailedResults;
    }

    /**
     * Get model
     */
    public AbstractDeepLearningModel getModel() {
        return model;
    }

    /**
     * Get dataset definition
     */
    public DatasetDefinition getDatasetDefinition() {
        return datasetDef;
    }

    /**
     * Get batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Print inference results summary
     */
    public static void printResultsSummary(InferenceResult result, AbstractDeepLearningModel model) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Inference Results Summary");
        System.out.println("=".repeat(80));
        System.out.println("Total objects classified: " + result.getTotalObjects());
        System.out.println("\nClass Distribution:");

        for (int classId : result.classDistribution.keySet()) {
            String className = model.getClassName(classId);
            int count = result.classDistribution.get(classId);
            double percentage = result.getClassPercentage(classId);

            System.out.printf("  %s (ID=%d): %d objects (%.1f%%)\n",
                            className, classId, count, percentage);
        }

        System.out.println("=".repeat(80));
    }
}
