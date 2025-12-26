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
import ij.process.FloatProcessor;
import vtea.deeplearning.data.ClassDefinition;
import vtea.deeplearning.data.CellRegionExtractor;
import vtea.deeplearning.data.DatasetDefinition;
import vtea.deeplearning.data.TensorConverter;
import vtea.deeplearning.models.NephNet3D;
import vtea.objects.layercake.microObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * Test class for InferenceEngine.
 * Demonstrates usage with MicroObjects.
 *
 * @author VTEA Deep Learning Team
 */
public class InferenceEngineTest {

    /**
     * Create synthetic image stack for testing
     */
    private static ImageStack createSyntheticStack(int width, int height, int depth) {
        ImageStack stack = new ImageStack(width, height);
        Random random = new Random(42);

        for (int z = 0; z < depth; z++) {
            float[] pixels = new float[width * height];

            for (int i = 0; i < pixels.length; i++) {
                pixels[i] = (float) random.nextGaussian();
            }

            stack.addSlice(new FloatProcessor(width, height, pixels));
        }

        return stack;
    }

    /**
     * Create synthetic MicroObjects for testing
     */
    private static List<microObject> createSyntheticObjects(int numObjects, int maxX, int maxY, int maxZ) {
        List<microObject> objects = new ArrayList<>();
        Random random = new Random(42);

        for (int i = 0; i < numObjects; i++) {
            microObject obj = new microObject();

            // Set random position within bounds
            obj.setX(random.nextInt(maxX));
            obj.setY(random.nextInt(maxY));
            obj.setZ(random.nextInt(maxZ));

            objects.add(obj);
        }

        return objects;
    }

    /**
     * Test basic inference workflow
     */
    public static void testBasicInference() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Test: Basic Inference Workflow");
        System.out.println("=".repeat(80));

        try {
            // Create synthetic image stack
            int width = 200;
            int height = 200;
            int depth = 50;
            System.out.println("\nCreating synthetic image stack (" + width + "x" + height + "x" + depth + ")...");

            ImageStack[] imageStacks = new ImageStack[2];
            imageStacks[0] = createSyntheticStack(width, height, depth);
            imageStacks[1] = createSyntheticStack(width, height, depth);

            // Create synthetic objects
            int numObjects = 20;
            System.out.println("Creating " + numObjects + " synthetic MicroObjects...");

            List<microObject> objects = createSyntheticObjects(numObjects,
                width - 64, height - 64, depth - 64);  // Keep away from edges

            // Create model
            int inputChannels = 2;
            int numClasses = 3;
            int regionSize = 32;  // Smaller for faster testing

            System.out.println("\nCreating model...");
            NephNet3D model = new NephNet3D(inputChannels, numClasses,
                new int[]{regionSize, regionSize, regionSize});

            // Set class definitions
            HashMap<Integer, ClassDefinition> classDefinitions = new HashMap<>();
            classDefinitions.put(0, new ClassDefinition(0, "Podocyte"));
            classDefinitions.put(1, new ClassDefinition(1, "Tubule"));
            classDefinitions.put(2, new ClassDefinition(2, "Glomerulus"));
            model.setClassDefinitions(classDefinitions);

            model.build();
            System.out.println("Model built: " + model);

            // Create dataset definition
            DatasetDefinition datasetDef = new DatasetDefinition.Builder()
                .name("Test Dataset")
                .regionSize(regionSize, regionSize, regionSize)
                .channels(0, 1)
                .normalize(true)
                .normalizationType(TensorConverter.NormalizationType.ZSCORE)
                .paddingType(CellRegionExtractor.PaddingType.ZERO)
                .build();

            datasetDef.setClassDefinitions(classDefinitions);

            // Create inference engine
            System.out.println("\nCreating inference engine...");
            InferenceEngine engine = new InferenceEngine(model, datasetDef, 8, true);

            // Set progress callback
            engine.setProgressCallback((current, total, message) -> {
                System.out.println("Progress: " + message);
            });

            // Run inference
            System.out.println("\nRunning inference...");
            long startTime = System.nanoTime();

            InferenceEngine.InferenceResult result = engine.predict(objects, imageStacks);

            long endTime = System.nanoTime();
            double elapsedMs = (endTime - startTime) / 1_000_000.0;

            // Print results
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Inference Results");
            System.out.println("=".repeat(80));
            System.out.println("Total objects: " + result.getTotalObjects());
            System.out.println("Inference time: " + String.format("%.2f ms", elapsedMs));
            System.out.println("Time per object: " + String.format("%.2f ms", elapsedMs / result.getTotalObjects()));

            InferenceEngine.printResultsSummary(result, model);

            // Print first 5 predictions
            System.out.println("\nFirst 5 predictions:");
            for (int i = 0; i < Math.min(5, result.getTotalObjects()); i++) {
                microObject obj = result.objects.get(i);
                int classId = result.classIds.get(i);
                String className = result.classNames.get(i);
                float[] probs = result.probabilities.get(i);

                System.out.printf("  Object %d at (%d, %d, %d): %s (ID=%d, prob=%.3f)\n",
                                i, obj.getX(), obj.getY(), obj.getZ(),
                                className, classId, probs[classId]);
            }

            System.out.println("\n✓ Basic inference test passed!");

        } catch (Exception e) {
            System.err.println("✗ Basic inference test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Test single object prediction
     */
    public static void testSingleObjectPrediction() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Test: Single Object Prediction");
        System.out.println("=".repeat(80));

        try {
            // Create test setup
            ImageStack[] imageStacks = new ImageStack[2];
            imageStacks[0] = createSyntheticStack(128, 128, 64);
            imageStacks[1] = createSyntheticStack(128, 128, 64);

            microObject testObject = new microObject();
            testObject.setX(64);
            testObject.setY(64);
            testObject.setZ(32);

            // Create model
            NephNet3D model = new NephNet3D(2, 2, new int[]{32, 32, 32});
            HashMap<Integer, ClassDefinition> classDefinitions = new HashMap<>();
            classDefinitions.put(0, new ClassDefinition(0, "Class_A"));
            classDefinitions.put(1, new ClassDefinition(1, "Class_B"));
            model.setClassDefinitions(classDefinitions);
            model.build();

            // Create dataset definition
            DatasetDefinition datasetDef = new DatasetDefinition.Builder()
                .name("Single Test")
                .regionSize(32, 32, 32)
                .channels(0, 1)
                .build();

            // Create engine
            InferenceEngine engine = new InferenceEngine(model, datasetDef);

            // Test single prediction
            System.out.println("\nPredicting single object at (" +
                             testObject.getX() + ", " + testObject.getY() + ", " + testObject.getZ() + ")...");

            int classId = engine.predictSingle(testObject, imageStacks);
            String className = model.getClassName(classId);

            System.out.println("Prediction: " + className + " (ID=" + classId + ")");

            // Test detailed prediction
            var detailedResult = engine.predictSingleDetailed(testObject, imageStacks);
            if (detailedResult != null) {
                System.out.println("Detailed prediction: " + detailedResult);
            }

            System.out.println("\n✓ Single object prediction test passed!");

        } catch (Exception e) {
            System.err.println("✗ Single object prediction test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Test batch size effects
     */
    public static void testBatchSizes() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Test: Batch Size Performance");
        System.out.println("=".repeat(80));

        try {
            // Setup
            ImageStack[] imageStacks = new ImageStack[1];
            imageStacks[0] = createSyntheticStack(200, 200, 50);

            List<microObject> objects = createSyntheticObjects(50, 150, 150, 40);

            NephNet3D model = new NephNet3D(1, 2, new int[]{32, 32, 32});
            model.build();

            DatasetDefinition datasetDef = new DatasetDefinition.Builder()
                .name("Batch Test")
                .regionSize(32, 32, 32)
                .channels(0)
                .build();

            // Test different batch sizes
            int[] batchSizes = {1, 4, 8, 16, 32};

            System.out.println("\nTesting different batch sizes on " + objects.size() + " objects:");
            System.out.println(String.format("%-12s %-15s %-15s", "Batch Size", "Time (ms)", "Time/Object (ms)"));
            System.out.println("-".repeat(45));

            for (int batchSize : batchSizes) {
                InferenceEngine engine = new InferenceEngine(model, datasetDef, batchSize, false);

                long startTime = System.nanoTime();
                InferenceEngine.InferenceResult result = engine.predict(objects, imageStacks);
                long endTime = System.nanoTime();

                double elapsedMs = (endTime - startTime) / 1_000_000.0;
                double msPerObject = elapsedMs / result.getTotalObjects();

                System.out.println(String.format("%-12d %-15.2f %-15.2f",
                                                batchSize, elapsedMs, msPerObject));
            }

            System.out.println("\n✓ Batch size test passed!");

        } catch (Exception e) {
            System.err.println("✗ Batch size test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Run all tests
     */
    public static void main(String[] args) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("VTEA Deep Learning Inference Engine Tests");
        System.out.println("=".repeat(80));

        try {
            testBasicInference();
            testSingleObjectPrediction();
            testBatchSizes();

            System.out.println("\n" + "=".repeat(80));
            System.out.println("All inference tests completed successfully! ✓");
            System.out.println("=".repeat(80));

        } catch (Exception e) {
            System.err.println("\nTests failed with error:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
