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
import org.bytedeco.pytorch.Tensor;
import static org.bytedeco.pytorch.global.torch.*;
import vtea.deeplearning.data.CellRegionExtractor;
import vtea.deeplearning.data.DatasetDefinition;
import vtea.deeplearning.data.TensorConverter;
import vtea.objects.layercake.microObject;

import java.util.*;

/**
 * DataLoader for batching and loading training/validation data.
 * Supports shuffling, augmentation, and efficient batching.
 *
 * @author VTEA Deep Learning Team
 */
public class DataLoader {

    private final List<TrainingSample> samples;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean augment;
    private final DatasetDefinition datasetDef;
    private final Random random;

    private int currentIndex;
    private List<Integer> indices;

    /**
     * Training sample containing region data and label
     */
    public static class TrainingSample {
        public final ImageStack[] region;  // Multi-channel region
        public final int label;
        public final microObject object;   // Original object (optional, for reference)

        public TrainingSample(ImageStack[] region, int label, microObject object) {
            this.region = region;
            this.label = label;
            this.object = object;
        }

        public TrainingSample(ImageStack[] region, int label) {
            this(region, label, null);
        }
    }

    /**
     * Batch containing tensors and labels
     */
    public static class Batch {
        public final Tensor data;      // [batch, channels, depth, height, width]
        public final Tensor labels;    // [batch]
        public final int size;         // Actual batch size (may be < batchSize for last batch)

        public Batch(Tensor data, Tensor labels, int size) {
            this.data = data;
            this.labels = labels;
            this.size = size;
        }

        public void close() {
            if (data != null) data.close();
            if (labels != null) labels.close();
        }
    }

    /**
     * Constructor
     */
    public DataLoader(List<TrainingSample> samples, DatasetDefinition datasetDef,
                     int batchSize, boolean shuffle, boolean augment) {
        this.samples = samples;
        this.datasetDef = datasetDef;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.augment = augment;
        this.random = new Random();

        this.currentIndex = 0;
        initializeIndices();
    }

    /**
     * Initialize indices for iteration
     */
    private void initializeIndices() {
        indices = new ArrayList<>();
        for (int i = 0; i < samples.size(); i++) {
            indices.add(i);
        }
        if (shuffle) {
            Collections.shuffle(indices, random);
        }
    }

    /**
     * Reset the data loader for a new epoch
     */
    public void reset() {
        currentIndex = 0;
        if (shuffle) {
            Collections.shuffle(indices, random);
        }
    }

    /**
     * Check if there are more batches
     */
    public boolean hasNext() {
        return currentIndex < samples.size();
    }

    /**
     * Get next batch
     */
    public Batch nextBatch() {
        if (!hasNext()) {
            throw new NoSuchElementException("No more batches available");
        }

        int actualBatchSize = Math.min(batchSize, samples.size() - currentIndex);
        List<ImageStack[]> batchRegions = new ArrayList<>();
        List<Integer> batchLabels = new ArrayList<>();

        for (int i = 0; i < actualBatchSize; i++) {
            int idx = indices.get(currentIndex + i);
            TrainingSample sample = samples.get(idx);

            ImageStack[] region = sample.region;

            // Apply augmentation if enabled
            if (augment) {
                region = applyAugmentation(region);
            }

            batchRegions.add(region);
            batchLabels.add(sample.label);
        }

        currentIndex += actualBatchSize;

        // Convert to tensors
        Tensor dataTensor = TensorConverter.batchRegionsToTensor(
            batchRegions,
            datasetDef.getChannels(),
            datasetDef.getNormalizationType()
        );

        // Convert labels to tensor
        long[] labelArray = new long[batchLabels.size()];
        for (int i = 0; i < batchLabels.size(); i++) {
            labelArray[i] = batchLabels.get(i);
        }
        Tensor labelTensor = from_blob(labelArray, new long[]{labelArray.length});

        return new Batch(dataTensor, labelTensor, actualBatchSize);
    }

    /**
     * Apply data augmentation to a region
     */
    private ImageStack[] applyAugmentation(ImageStack[] region) {
        // Create augmented copies
        ImageStack[] augmented = new ImageStack[region.length];

        for (int c = 0; c < region.length; c++) {
            ImageStack stack = region[c];
            ImageStack augStack = stack.duplicate();

            // Random flips (50% chance for each axis)
            if (random.nextBoolean()) {
                augStack = flipX(augStack);
            }
            if (random.nextBoolean()) {
                augStack = flipY(augStack);
            }
            if (random.nextBoolean()) {
                augStack = flipZ(augStack);
            }

            // Random 90-degree rotations around Z-axis
            int rotations = random.nextInt(4);  // 0, 1, 2, or 3 rotations
            for (int r = 0; r < rotations; r++) {
                augStack = rotate90Z(augStack);
            }

            // Random intensity scaling (0.9 to 1.1)
            if (random.nextBoolean()) {
                double scale = 0.9 + random.nextDouble() * 0.2;
                augStack = scaleIntensity(augStack, scale);
            }

            // Random noise addition (Gaussian noise)
            if (random.nextBoolean()) {
                double noiseStd = 0.01 + random.nextDouble() * 0.02;
                augStack = addGaussianNoise(augStack, noiseStd);
            }

            augmented[c] = augStack;
        }

        return augmented;
    }

    /**
     * Flip image stack along X-axis
     */
    private ImageStack flipX(ImageStack stack) {
        int width = stack.getWidth();
        int height = stack.getHeight();
        int depth = stack.getSize();

        ImageStack flipped = new ImageStack(width, height);

        for (int z = 1; z <= depth; z++) {
            float[] pixels = (float[]) stack.getProcessor(z).getPixels();
            float[] flippedPixels = new float[width * height];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    flippedPixels[y * width + x] = pixels[y * width + (width - 1 - x)];
                }
            }

            flipped.addSlice(new ij.process.FloatProcessor(width, height, flippedPixels));
        }

        return flipped;
    }

    /**
     * Flip image stack along Y-axis
     */
    private ImageStack flipY(ImageStack stack) {
        int width = stack.getWidth();
        int height = stack.getHeight();
        int depth = stack.getSize();

        ImageStack flipped = new ImageStack(width, height);

        for (int z = 1; z <= depth; z++) {
            float[] pixels = (float[]) stack.getProcessor(z).getPixels();
            float[] flippedPixels = new float[width * height];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    flippedPixels[y * width + x] = pixels[(height - 1 - y) * width + x];
                }
            }

            flipped.addSlice(new ij.process.FloatProcessor(width, height, flippedPixels));
        }

        return flipped;
    }

    /**
     * Flip image stack along Z-axis
     */
    private ImageStack flipZ(ImageStack stack) {
        int width = stack.getWidth();
        int height = stack.getHeight();
        int depth = stack.getSize();

        ImageStack flipped = new ImageStack(width, height);

        for (int z = depth; z >= 1; z--) {
            flipped.addSlice(stack.getProcessor(z).duplicate());
        }

        return flipped;
    }

    /**
     * Rotate image stack 90 degrees around Z-axis
     */
    private ImageStack rotate90Z(ImageStack stack) {
        int width = stack.getWidth();
        int height = stack.getHeight();
        int depth = stack.getSize();

        ImageStack rotated = new ImageStack(height, width);  // Dimensions swap

        for (int z = 1; z <= depth; z++) {
            float[] pixels = (float[]) stack.getProcessor(z).getPixels();
            float[] rotatedPixels = new float[width * height];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // 90-degree clockwise rotation: (x,y) -> (height-1-y, x)
                    int newX = height - 1 - y;
                    int newY = x;
                    rotatedPixels[newY * height + newX] = pixels[y * width + x];
                }
            }

            rotated.addSlice(new ij.process.FloatProcessor(height, width, rotatedPixels));
        }

        return rotated;
    }

    /**
     * Scale intensity values
     */
    private ImageStack scaleIntensity(ImageStack stack, double scale) {
        int width = stack.getWidth();
        int height = stack.getHeight();
        int depth = stack.getSize();

        ImageStack scaled = new ImageStack(width, height);

        for (int z = 1; z <= depth; z++) {
            float[] pixels = (float[]) stack.getProcessor(z).getPixels();
            float[] scaledPixels = new float[pixels.length];

            for (int i = 0; i < pixels.length; i++) {
                scaledPixels[i] = (float) (pixels[i] * scale);
            }

            scaled.addSlice(new ij.process.FloatProcessor(width, height, scaledPixels));
        }

        return scaled;
    }

    /**
     * Add Gaussian noise
     */
    private ImageStack addGaussianNoise(ImageStack stack, double stdDev) {
        int width = stack.getWidth();
        int height = stack.getHeight();
        int depth = stack.getSize();

        ImageStack noisy = new ImageStack(width, height);

        for (int z = 1; z <= depth; z++) {
            float[] pixels = (float[]) stack.getProcessor(z).getPixels();
            float[] noisyPixels = new float[pixels.length];

            for (int i = 0; i < pixels.length; i++) {
                double noise = random.nextGaussian() * stdDev;
                noisyPixels[i] = (float) (pixels[i] + noise);
            }

            noisy.addSlice(new ij.process.FloatProcessor(width, height, noisyPixels));
        }

        return noisy;
    }

    /**
     * Get total number of samples
     */
    public int getNumSamples() {
        return samples.size();
    }

    /**
     * Get number of batches per epoch
     */
    public int getNumBatches() {
        return (int) Math.ceil((double) samples.size() / batchSize);
    }

    /**
     * Get current batch index
     */
    public int getCurrentBatchIndex() {
        return currentIndex / batchSize;
    }

    /**
     * Split samples into train and validation sets
     */
    public static DataLoaderSplit trainValidationSplit(List<TrainingSample> allSamples,
                                                       DatasetDefinition datasetDef,
                                                       double validationSplit,
                                                       int batchSize,
                                                       boolean augmentTrain) {
        if (validationSplit < 0 || validationSplit >= 1) {
            throw new IllegalArgumentException("Validation split must be in [0, 1)");
        }

        // Shuffle all samples
        List<TrainingSample> shuffled = new ArrayList<>(allSamples);
        Collections.shuffle(shuffled);

        // Split
        int validationSize = (int) (allSamples.size() * validationSplit);
        int trainSize = allSamples.size() - validationSize;

        List<TrainingSample> trainSamples = shuffled.subList(0, trainSize);
        List<TrainingSample> validationSamples = shuffled.subList(trainSize, allSamples.size());

        // Create data loaders
        DataLoader trainLoader = new DataLoader(trainSamples, datasetDef, batchSize, true, augmentTrain);
        DataLoader validationLoader = new DataLoader(validationSamples, datasetDef, batchSize, false, false);

        return new DataLoaderSplit(trainLoader, validationLoader);
    }

    /**
     * Container for train/validation split
     */
    public static class DataLoaderSplit {
        public final DataLoader trainLoader;
        public final DataLoader validationLoader;

        public DataLoaderSplit(DataLoader trainLoader, DataLoader validationLoader) {
            this.trainLoader = trainLoader;
            this.validationLoader = validationLoader;
        }
    }

    /**
     * Create samples from MicroObjects
     */
    public static List<TrainingSample> createSamples(List<microObject> objects,
                                                     ImageStack[] imageStacks,
                                                     Map<microObject, Integer> labels,
                                                     DatasetDefinition datasetDef) {
        List<TrainingSample> samples = new ArrayList<>();

        for (microObject object : objects) {
            Integer label = labels.get(object);
            if (label == null) {
                continue;  // Skip unlabeled objects
            }

            try {
                ImageStack[] region = CellRegionExtractor.extractRegion(
                    object,
                    imageStacks,
                    datasetDef.getRegionSize(),
                    datasetDef.getChannels(),
                    datasetDef.getPaddingType()
                );

                samples.add(new TrainingSample(region, label, object));

            } catch (Exception e) {
                System.err.println("Failed to extract region for object: " + e.getMessage());
            }
        }

        return samples;
    }

    /**
     * Get class distribution in the dataset
     */
    public Map<Integer, Integer> getClassDistribution() {
        Map<Integer, Integer> distribution = new HashMap<>();

        for (TrainingSample sample : samples) {
            distribution.put(sample.label, distribution.getOrDefault(sample.label, 0) + 1);
        }

        return distribution;
    }

    /**
     * Calculate class weights for imbalanced datasets
     */
    public double[] calculateClassWeights(int numClasses) {
        Map<Integer, Integer> distribution = getClassDistribution();

        double[] weights = new double[numClasses];
        int totalSamples = samples.size();

        for (int i = 0; i < numClasses; i++) {
            int classCount = distribution.getOrDefault(i, 0);
            if (classCount > 0) {
                weights[i] = (double) totalSamples / (numClasses * classCount);
            } else {
                weights[i] = 0;
            }
        }

        return weights;
    }

    @Override
    public String toString() {
        return String.format("DataLoader[samples=%d, batches=%d, batch_size=%d, shuffle=%s, augment=%s]",
                           samples.size(), getNumBatches(), batchSize, shuffle, augment);
    }
}
