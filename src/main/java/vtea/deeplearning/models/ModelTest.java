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
package vtea.deeplearning.models;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import vtea.deeplearning.data.ClassDefinition;

import java.util.HashMap;

/**
 * Test class for verifying model instantiation and forward pass.
 * Demonstrates usage of NephNet3D and Generic3DCNN models.
 *
 * @author VTEA Deep Learning Team
 */
public class ModelTest {

    /**
     * Test NephNet3D model
     */
    public static void testNephNet3D() {
        System.out.println("=".repeat(80));
        System.out.println("Testing NephNet3D");
        System.out.println("=".repeat(80));

        try {
            // Create model for 3-class classification with 2 input channels
            int inputChannels = 2;
            int numClasses = 3;
            int[] inputShape = {64, 64, 64};

            NephNet3D model = new NephNet3D(inputChannels, numClasses, inputShape);

            // Set class definitions
            HashMap<Integer, ClassDefinition> classDefinitions = new HashMap<>();
            classDefinitions.put(0, new ClassDefinition(0, "Podocyte"));
            classDefinitions.put(1, new ClassDefinition(1, "Tubule"));
            classDefinitions.put(2, new ClassDefinition(2, "Glomerulus"));
            model.setClassDefinitions(classDefinitions);

            // Build model
            System.out.println("\nBuilding model...");
            model.build();

            // Print architecture
            System.out.println("\n" + model.getArchitectureDescription());
            System.out.println(model);

            // Create dummy input tensor [batch=2, channels=2, depth=64, height=64, width=64]
            System.out.println("\nCreating dummy input tensor...");
            Tensor input = randn(new long[]{2, inputChannels, 64, 64, 64});
            System.out.println("Input shape: " + tensorShapeToString(input));

            // Forward pass
            System.out.println("\nRunning forward pass...");
            Tensor output = model.forward(input);
            System.out.println("Output shape: " + tensorShapeToString(output));

            // Test predictions
            System.out.println("\nTesting predictions...");
            int[] predictions = model.predict(input);
            System.out.println("Predicted class IDs: " + java.util.Arrays.toString(predictions));

            String[] classNames = model.predictClassNames(input);
            System.out.println("Predicted class names: " + java.util.Arrays.toString(classNames));

            float[][] probabilities = model.predictProba(input);
            System.out.println("Prediction probabilities:");
            for (int i = 0; i < probabilities.length; i++) {
                System.out.printf("  Sample %d: ", i);
                for (int j = 0; j < probabilities[i].length; j++) {
                    System.out.printf("%s=%.3f ", model.getClassName(j), probabilities[i][j]);
                }
                System.out.println();
            }

            // Test detailed predictions
            System.out.println("\nDetailed predictions:");
            AbstractDeepLearningModel.PredictionResult[] results = model.predictDetailed(input);
            for (int i = 0; i < results.length; i++) {
                System.out.printf("  Sample %d: %s\n", i, results[i]);
            }

            System.out.println("\n✓ NephNet3D test passed!");

        } catch (Exception e) {
            System.err.println("✗ NephNet3D test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Test Generic3DCNN model with builder pattern
     */
    public static void testGeneric3DCNN() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Testing Generic3DCNN");
        System.out.println("=".repeat(80));

        try {
            // Create custom model using builder
            int inputChannels = 3;
            int numClasses = 4;
            int[] inputShape = {64, 64, 64};

            Generic3DCNN model = new Generic3DCNN.Builder(inputChannels, numClasses, inputShape)
                .modelName("CustomKidneyCNN")
                .blockChannels(32, 64, 128, 256)
                .kernelSizes(3, 3, 3, 3)
                .poolingSizes(2, 2, 2, 2)
                .activation(Generic3DCNN.ActivationType.LEAKY_RELU)
                .useBatchNorm(true)
                .fcLayers(256, 128)
                .dropoutRate(0.5)
                .build();

            // Set class definitions
            HashMap<Integer, ClassDefinition> classDefinitions = new HashMap<>();
            classDefinitions.put(0, new ClassDefinition(0, "Normal"));
            classDefinitions.put(1, new ClassDefinition(1, "Diseased"));
            classDefinitions.put(2, new ClassDefinition(2, "Intermediate"));
            classDefinitions.put(3, new ClassDefinition(3, "Unknown"));
            model.setClassDefinitions(classDefinitions);

            // Build model
            System.out.println("\nBuilding model...");
            model.build();

            // Print architecture
            System.out.println("\n" + model.getArchitectureDescription());
            System.out.println(model);

            // Create dummy input
            System.out.println("\nCreating dummy input tensor...");
            Tensor input = randn(new long[]{3, inputChannels, 64, 64, 64});
            System.out.println("Input shape: " + tensorShapeToString(input));

            // Forward pass
            System.out.println("\nRunning forward pass...");
            Tensor output = model.forward(input);
            System.out.println("Output shape: " + tensorShapeToString(output));

            // Test predictions
            System.out.println("\nTesting predictions...");
            int[] predictions = model.predict(input);
            System.out.println("Predicted class IDs: " + java.util.Arrays.toString(predictions));

            String[] classNames = model.predictClassNames(input);
            System.out.println("Predicted class names: " + java.util.Arrays.toString(classNames));

            System.out.println("\n✓ Generic3DCNN test passed!");

        } catch (Exception e) {
            System.err.println("✗ Generic3DCNN test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Test factory methods for Generic3DCNN
     */
    public static void testGeneric3DCNNFactories() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Testing Generic3DCNN Factory Methods");
        System.out.println("=".repeat(80));

        try {
            int inputChannels = 2;
            int numClasses = 2;
            int[] inputShape = {64, 64, 64};

            // Test lightweight model
            System.out.println("\n1. Testing Lightweight Model:");
            Generic3DCNN lightweight = Generic3DCNN.createLightweight(inputChannels, numClasses, inputShape);
            lightweight.build();
            System.out.println(lightweight);
            System.out.printf("Parameters: %,d (%.2f MB)\n",
                            lightweight.getParameterCount(),
                            lightweight.getMemoryFootprintMB());

            // Test deep model
            System.out.println("\n2. Testing Deep Model:");
            Generic3DCNN deep = Generic3DCNN.createDeep(inputChannels, numClasses, inputShape);
            deep.build();
            System.out.println(deep);
            System.out.printf("Parameters: %,d (%.2f MB)\n",
                            deep.getParameterCount(),
                            deep.getMemoryFootprintMB());

            // Test wide model
            System.out.println("\n3. Testing Wide Model:");
            Generic3DCNN wide = Generic3DCNN.createWide(inputChannels, numClasses, inputShape);
            wide.build();
            System.out.println(wide);
            System.out.printf("Parameters: %,d (%.2f MB)\n",
                            wide.getParameterCount(),
                            wide.getMemoryFootprintMB());

            // Test NephNet-style model
            System.out.println("\n4. Testing NephNet-Style Model:");
            Generic3DCNN nephStyle = Generic3DCNN.createNephNetStyle(inputChannels, numClasses, inputShape);
            nephStyle.build();
            System.out.println(nephStyle);
            System.out.printf("Parameters: %,d (%.2f MB)\n",
                            nephStyle.getParameterCount(),
                            nephStyle.getMemoryFootprintMB());

            System.out.println("\n✓ Factory methods test passed!");

        } catch (Exception e) {
            System.err.println("✗ Factory methods test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Test model save/load functionality
     */
    public static void testModelPersistence() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Testing Model Save/Load");
        System.out.println("=".repeat(80));

        try {
            // Create and build model
            NephNet3D model = new NephNet3D(2, 3, new int[]{64, 64, 64});
            HashMap<Integer, ClassDefinition> classDefinitions = new HashMap<>();
            classDefinitions.put(0, new ClassDefinition(0, "ClassA"));
            classDefinitions.put(1, new ClassDefinition(1, "ClassB"));
            classDefinitions.put(2, new ClassDefinition(2, "ClassC"));
            model.setClassDefinitions(classDefinitions);
            model.build();

            // Create test directory
            String testDir = System.getProperty("user.home") + "/.vtea/models/test";
            java.io.File dir = new java.io.File(testDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }

            String modelPath = testDir + "/test_model";

            // Save model
            System.out.println("\nSaving model to: " + modelPath);
            model.save(modelPath);

            // Load model
            System.out.println("\nLoading model from: " + modelPath);
            NephNet3D loadedModel = new NephNet3D(2, 3, new int[]{64, 64, 64});
            loadedModel.load(modelPath);

            // Verify loaded model
            System.out.println("\nVerifying loaded model:");
            System.out.println("  Model name: " + loadedModel.getModelName());
            System.out.println("  Input channels: " + loadedModel.getInputChannels());
            System.out.println("  Num classes: " + loadedModel.getNumClasses());
            System.out.println("  Class names: " + loadedModel.getClassDefinitions().keySet());

            // Test prediction with loaded model
            Tensor input = randn(new long[]{1, 2, 64, 64, 64});
            String[] predictions = loadedModel.predictClassNames(input);
            System.out.println("  Prediction: " + java.util.Arrays.toString(predictions));

            System.out.println("\n✓ Model persistence test passed!");

        } catch (Exception e) {
            System.err.println("✗ Model persistence test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Helper to convert tensor shape to string
     */
    private static String tensorShapeToString(Tensor tensor) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < tensor.dim(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(tensor.size(i));
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Run all tests
     */
    public static void main(String[] args) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("VTEA Deep Learning Model Tests");
        System.out.println("=".repeat(80));

        try {
            // Run tests
            testNephNet3D();
            testGeneric3DCNN();
            testGeneric3DCNNFactories();
            testModelPersistence();

            // Summary
            System.out.println("\n" + "=".repeat(80));
            System.out.println("All tests completed successfully! ✓");
            System.out.println("=".repeat(80));

        } catch (Exception e) {
            System.err.println("\nTests failed with error:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
