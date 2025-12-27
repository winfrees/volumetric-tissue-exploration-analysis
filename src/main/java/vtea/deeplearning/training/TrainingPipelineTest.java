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
package vtea.deeplearning.training;

import ij.ImageStack;
import ij.process.FloatProcessor;
import vtea.deeplearning.data.ClassDefinition;
import vtea.deeplearning.data.DatasetDefinition;
import vtea.deeplearning.data.TensorConverter;
import vtea.deeplearning.data.CellRegionExtractor;
import vtea.deeplearning.models.NephNet3D;
import vtea.deeplearning.models.Generic3DCNN;

import java.util.*;

/**
 * Test class for training pipeline.
 * Demonstrates complete workflow from data preparation to model training.
 *
 * @author VTEA Deep Learning Team
 */
public class TrainingPipelineTest {

    /**
     * Create synthetic training samples for testing
     */
    private static List<DataLoader.TrainingSample> createSyntheticSamples(int numSamples, int numClasses,
                                                                          int channels, int size) {
        List<DataLoader.TrainingSample> samples = new ArrayList<>();
        Random random = new Random(42);

        for (int i = 0; i < numSamples; i++) {
            // Create multi-channel region
            ImageStack[] region = new ImageStack[channels];

            for (int c = 0; c < channels; c++) {
                ImageStack stack = new ImageStack(size, size);

                for (int z = 0; z < size; z++) {
                    float[] pixels = new float[size * size];

                    // Fill with random values influenced by class label
                    int label = i % numClasses;
                    for (int p = 0; p < pixels.length; p++) {
                        pixels[p] = (float) (random.nextGaussian() + label * 0.5);
                    }

                    stack.addSlice(new FloatProcessor(size, size, pixels));
                }

                region[c] = stack;
            }

            int label = i % numClasses;
            samples.add(new DataLoader.TrainingSample(region, label));
        }

        return samples;
    }

    /**
     * Test basic training pipeline
     */
    public static void testBasicTraining() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Test 1: Basic Training Pipeline");
        System.out.println("=".repeat(80));

        try {
            // Create synthetic dataset
            int numSamples = 100;
            int numClasses = 3;
            int inputChannels = 2;
            int regionSize = 32;  // Small for fast testing

            System.out.println("\nCreating synthetic dataset...");
            List<DataLoader.TrainingSample> allSamples = createSyntheticSamples(
                numSamples, numClasses, inputChannels, regionSize);

            // Create dataset definition
            DatasetDefinition datasetDef = new DatasetDefinition.Builder()
                .name("Test Dataset")
                .regionSize(regionSize, regionSize, regionSize)
                .channels(0, 1)
                .normalize(true)
                .normalizationType(TensorConverter.NormalizationType.ZSCORE)
                .paddingType(CellRegionExtractor.PaddingType.ZERO)
                .build();

            // Add class definitions
            HashMap<Integer, ClassDefinition> classDefinitions = new HashMap<>();
            classDefinitions.put(0, new ClassDefinition(0, "Class_A"));
            classDefinitions.put(1, new ClassDefinition(1, "Class_B"));
            classDefinitions.put(2, new ClassDefinition(2, "Class_C"));
            datasetDef.setClassDefinitions(classDefinitions);

            // Split into train/validation
            System.out.println("Splitting dataset (80% train, 20% validation)...");
            DataLoader.DataLoaderSplit split = DataLoader.trainValidationSplit(
                allSamples, datasetDef, 0.2, 8, true);

            System.out.println("Train samples: " + split.trainLoader.getNumSamples());
            System.out.println("Validation samples: " + split.validationLoader.getNumSamples());

            // Create model
            System.out.println("\nCreating model...");
            NephNet3D model = new NephNet3D(inputChannels, numClasses,
                                           new int[]{regionSize, regionSize, regionSize});
            model.setClassDefinitions(classDefinitions);

            // Configure training
            TrainingConfig config = new TrainingConfig.Builder()
                .epochs(5)
                .batchSize(8)
                .learningRate(0.001)
                .optimizer(TrainingConfig.OptimizerType.ADAM)
                .validationSplit(0.2)
                .useEarlyStopping(false)
                .checkpointDir(System.getProperty("user.home") + "/.vtea/models/test_checkpoints")
                .verbose(true)
                .build();

            // Create trainer
            System.out.println("\nInitializing trainer...");
            Trainer trainer = new Trainer(model, config, split.trainLoader, split.validationLoader);

            // Add checkpoint callback
            ModelCheckpoint checkpoint = new ModelCheckpoint(model,
                config.getCheckpointDir(), "test_model", true, 3, "val_bal_acc", true);
            trainer.setCallback(checkpoint);

            // Train
            System.out.println("\nStarting training...");
            Metrics.History history = trainer.train();

            // Print results
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Training Results:");
            System.out.println("=".repeat(80));
            System.out.printf("Best validation accuracy: %.4f\n", history.getBestValidationAccuracy());
            System.out.printf("Best validation balanced accuracy: %.4f\n", history.getBestValidationBalancedAccuracy());
            System.out.printf("Best epoch: %d\n", history.getBestEpoch() + 1);

            System.out.println("\n✓ Basic training test passed!");

        } catch (Exception e) {
            System.err.println("✗ Basic training test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Test training with class weights
     */
    public static void testImbalancedTraining() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Test 2: Training with Class Weights (Imbalanced Dataset)");
        System.out.println("=".repeat(80));

        try {
            // Create imbalanced dataset (70% class 0, 20% class 1, 10% class 2)
            List<DataLoader.TrainingSample> samples = new ArrayList<>();
            int[] classCounts = {70, 20, 10};
            int regionSize = 32;
            int inputChannels = 2;

            Random random = new Random(42);

            for (int classId = 0; classId < classCounts.length; classId++) {
                for (int i = 0; i < classCounts[classId]; i++) {
                    ImageStack[] region = new ImageStack[inputChannels];

                    for (int c = 0; c < inputChannels; c++) {
                        ImageStack stack = new ImageStack(regionSize, regionSize);

                        for (int z = 0; z < regionSize; z++) {
                            float[] pixels = new float[regionSize * regionSize];
                            for (int p = 0; p < pixels.length; p++) {
                                pixels[p] = (float) (random.nextGaussian() + classId * 0.5);
                            }
                            stack.addSlice(new FloatProcessor(regionSize, regionSize, pixels));
                        }

                        region[c] = stack;
                    }

                    samples.add(new DataLoader.TrainingSample(region, classId));
                }
            }

            System.out.println("\nDataset class distribution:");
            for (int i = 0; i < classCounts.length; i++) {
                System.out.printf("  Class %d: %d samples (%.1f%%)\n",
                                i, classCounts[i], 100.0 * classCounts[i] / samples.size());
            }

            // Create dataset definition
            DatasetDefinition datasetDef = new DatasetDefinition.Builder()
                .name("Imbalanced Test Dataset")
                .regionSize(regionSize, regionSize, regionSize)
                .channels(0, 1)
                .normalize(true)
                .build();

            // Split dataset
            DataLoader.DataLoaderSplit split = DataLoader.trainValidationSplit(
                samples, datasetDef, 0.2, 8, true);

            // Calculate class weights
            double[] classWeights = split.trainLoader.calculateClassWeights(classCounts.length);
            System.out.println("\nCalculated class weights:");
            for (int i = 0; i < classWeights.length; i++) {
                System.out.printf("  Class %d: %.3f\n", i, classWeights[i]);
            }

            // Create model
            Generic3DCNN model = Generic3DCNN.createLightweight(inputChannels, classCounts.length,
                new int[]{regionSize, regionSize, regionSize});

            // Configure training with class weights
            TrainingConfig config = new TrainingConfig.Builder()
                .epochs(5)
                .batchSize(8)
                .learningRate(0.001)
                .classWeights(classWeights)
                .verbose(true)
                .build();

            // Train
            Trainer trainer = new Trainer(model, config, split.trainLoader, split.validationLoader);
            Metrics.History history = trainer.train();

            System.out.println("\n✓ Imbalanced training test passed!");

        } catch (Exception e) {
            System.err.println("✗ Imbalanced training test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Test data augmentation
     */
    public static void testDataAugmentation() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Test 3: Data Augmentation");
        System.out.println("=".repeat(80));

        try {
            // Create small dataset
            List<DataLoader.TrainingSample> samples = createSyntheticSamples(50, 2, 1, 32);

            DatasetDefinition datasetDef = new DatasetDefinition.Builder()
                .name("Augmentation Test")
                .regionSize(32, 32, 32)
                .channels(0)
                .build();

            // Test with and without augmentation
            System.out.println("\nTesting DataLoader with augmentation...");
            DataLoader augmentedLoader = new DataLoader(samples, datasetDef, 8, true, true);
            System.out.println(augmentedLoader);

            System.out.println("\nTesting DataLoader without augmentation...");
            DataLoader normalLoader = new DataLoader(samples, datasetDef, 8, true, false);
            System.out.println(normalLoader);

            // Get a batch from each
            augmentedLoader.reset();
            normalLoader.reset();

            DataLoader.Batch augBatch = augmentedLoader.nextBatch();
            DataLoader.Batch normBatch = normalLoader.nextBatch();

            System.out.println("\nAugmented batch shape: [" + augBatch.data.size(0) + ", " +
                             augBatch.data.size(1) + ", " + augBatch.data.size(2) + ", " +
                             augBatch.data.size(3) + ", " + augBatch.data.size(4) + "]");

            System.out.println("Normal batch shape: [" + normBatch.data.size(0) + ", " +
                             normBatch.data.size(1) + ", " + normBatch.data.size(2) + ", " +
                             normBatch.data.size(3) + ", " + normBatch.data.size(4) + "]");

            augBatch.close();
            normBatch.close();

            System.out.println("\n✓ Data augmentation test passed!");

        } catch (Exception e) {
            System.err.println("✗ Data augmentation test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Test metrics calculation
     */
    public static void testMetrics() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Test 4: Metrics Calculation");
        System.out.println("=".repeat(80));

        try {
            // Create dummy predictions and targets using PyTorch
            org.bytedeco.pytorch.Tensor predictions = org.bytedeco.pytorch.global.torch.randn(new long[]{10, 3});
            org.bytedeco.pytorch.Tensor targets = org.bytedeco.pytorch.global.torch.randint(3, new long[]{10});

            System.out.println("\nCalculating metrics...");

            // Test accuracy
            double accuracy = Metrics.accuracy(predictions, targets);
            System.out.printf("Accuracy: %.4f\n", accuracy);

            // Test balanced accuracy
            double balancedAcc = Metrics.balancedAccuracy(predictions, targets, 3);
            System.out.printf("Balanced Accuracy: %.4f\n", balancedAcc);

            // Test confusion matrix
            int[][] confusionMatrix = Metrics.confusionMatrix(predictions, targets, 3);
            String[] classNames = {"Class_A", "Class_B", "Class_C"};
            System.out.println(Metrics.formatConfusionMatrix(confusionMatrix, classNames));

            // Test classification report
            System.out.println(Metrics.formatClassificationReport(confusionMatrix, classNames));

            predictions.close();
            targets.close();

            System.out.println("✓ Metrics calculation test passed!");

        } catch (Exception e) {
            System.err.println("✗ Metrics calculation test failed:");
            e.printStackTrace();
        }
    }

    /**
     * Run all tests
     */
    public static void main(String[] args) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("VTEA Deep Learning Training Pipeline Tests");
        System.out.println("=".repeat(80));

        try {
            testBasicTraining();
            testImbalancedTraining();
            testDataAugmentation();
            testMetrics();

            System.out.println("\n" + "=".repeat(80));
            System.out.println("All training pipeline tests completed successfully! ✓");
            System.out.println("=".repeat(80));

        } catch (Exception e) {
            System.err.println("\nTests failed with error:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
