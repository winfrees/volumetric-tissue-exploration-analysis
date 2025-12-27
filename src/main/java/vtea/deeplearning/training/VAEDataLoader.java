package vtea.deeplearning.training;

import ij.ImageStack;
import org.bytedeco.pytorch.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import vtea.deeplearning.data.CellRegionExtractor;
import vtea.deeplearning.data.TensorConverter;
import vteaobjects.MicroObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * DataLoader for batching and shuffling cell data for VAE training.
 *
 * <p>Integrates with VTEA's MicroObject and ImageStack data structures,
 * following VTEA patterns for data processing.</p>
 *
 * <p>Features:</p>
 * <ul>
 *   <li>Batch creation from MicroObject lists</li>
 *   <li>Shuffling with configurable random seed</li>
 *   <li>Optional data augmentation (rotation, flip, noise)</li>
 *   <li>Multi-channel support</li>
 *   <li>Progress tracking compatible with VTEA ProgressListener</li>
 * </ul>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class VAEDataLoader {

    private static final Logger logger = LoggerFactory.getLogger(DataLoader.class);

    private final List<MicroObject> cells;
    private final ImageStack[] imageStacks;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean useAugmentation;
    private final Random random;

    private final TensorConverter converter;
    private final CellRegionExtractor extractor;

    private List<Batch> batches;
    private int currentBatchIndex;
    private int epochCount;

    /**
     * Creates a DataLoader.
     *
     * @param cells List of MicroObjects (cells to process)
     * @param imageStacks Source image data (multi-channel)
     * @param batchSize Batch size
     * @param shuffle Whether to shuffle data each epoch
     * @param useAugmentation Whether to apply data augmentation
     * @param regionSize Size of extracted regions (e.g., 64 for 64³)
     * @param normalization Normalization strategy
     * @param randomSeed Random seed for reproducibility
     */
    public VAEDataLoader(List<MicroObject> cells,
                     ImageStack[] imageStacks,
                     int batchSize,
                     boolean shuffle,
                     boolean useAugmentation,
                     int regionSize,
                     TensorConverter.NormalizationType normalization,
                     long randomSeed) {

        if (cells == null || cells.isEmpty()) {
            throw new IllegalArgumentException("Cells list is null or empty");
        }
        if (imageStacks == null || imageStacks.length == 0) {
            throw new IllegalArgumentException("ImageStacks is null or empty");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }

        this.cells = new ArrayList<>(cells);
        this.imageStacks = imageStacks;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.useAugmentation = useAugmentation;
        this.random = new Random(randomSeed);

        this.converter = new TensorConverter(normalization, false);
        this.extractor = new CellRegionExtractor(regionSize,
                        CellRegionExtractor.PaddingType.MIRROR);

        this.currentBatchIndex = 0;
        this.epochCount = 0;

        logger.info("DataLoader created: {} cells, batch_size={}, shuffle={}, " +
                   "augmentation={}, region_size={}",
                   cells.size(), batchSize, shuffle, useAugmentation, regionSize);

        // Pre-compute batches
        createBatches();
    }

    /**
     * Creates batches from the cell list.
     */
    private void createBatches() {
        List<MicroObject> workingCells = new ArrayList<>(cells);

        if (shuffle) {
            Collections.shuffle(workingCells, random);
            logger.debug("Shuffled {} cells for epoch {}", workingCells.size(), epochCount);
        }

        batches = new ArrayList<>();
        int numBatches = (int) Math.ceil((double) workingCells.size() / batchSize);

        for (int i = 0; i < numBatches; i++) {
            int startIdx = i * batchSize;
            int endIdx = Math.min(startIdx + batchSize, workingCells.size());

            List<MicroObject> batchCells = workingCells.subList(startIdx, endIdx);
            batches.add(new Batch(batchCells, i));
        }

        logger.debug("Created {} batches (last batch size: {})",
                    batches.size(),
                    batches.get(batches.size() - 1).getCells().size());
    }

    /**
     * Gets the next batch.
     *
     * @return Next batch with tensor data
     */
    public Batch nextBatch() {
        if (!hasNext()) {
            throw new IllegalStateException("No more batches. Call reset() to start new epoch.");
        }

        Batch batch = batches.get(currentBatchIndex);

        // Process batch: extract regions and convert to tensors
        List<Tensor> tensors = new ArrayList<>();

        for (MicroObject cell : batch.getCells()) {
            try {
                // Extract region
                ImageStack[] regions = extractor.extractRegion(cell, imageStacks);

                // Convert to tensor
                Tensor tensor = (regions.length == 1) ?
                               converter.imageStackToTensor(regions[0]) :
                               converter.imageStacksToTensor(regions);

                // Apply augmentation if enabled
                if (useAugmentation) {
                    tensor = augment(tensor);
                }

                tensors.add(tensor);

            } catch (Exception e) {
                logger.warn("Failed to process cell at index {}: {}",
                           batch.getIndex() * batchSize + tensors.size(),
                           e.getMessage());
                // Skip problematic cells
            }
        }

        // Stack tensors into batch
        Tensor batchTensor = stackTensors(tensors);
        batch.setData(batchTensor);

        currentBatchIndex++;

        logger.debug("Loaded batch {}/{} with {} samples",
                    currentBatchIndex, batches.size(), tensors.size());

        return batch;
    }

    /**
     * Checks if more batches are available.
     *
     * @return true if more batches available
     */
    public boolean hasNext() {
        return currentBatchIndex < batches.size();
    }

    /**
     * Resets iterator for new epoch.
     */
    public void reset() {
        currentBatchIndex = 0;
        epochCount++;

        if (shuffle) {
            createBatches(); // Reshuffle for new epoch
        }

        logger.debug("DataLoader reset for epoch {}", epochCount);
    }

    /**
     * Gets total number of batches.
     *
     * @return Number of batches
     */
    public int size() {
        return batches.size();
    }

    /**
     * Gets total number of samples.
     *
     * @return Total cells
     */
    public int getTotalSamples() {
        return cells.size();
    }

    /**
     * Gets current epoch count.
     *
     * @return Epoch number
     */
    public int getEpochCount() {
        return epochCount;
    }

    /**
     * Applies data augmentation to a tensor.
     *
     * @param tensor Input tensor [1, C, D, H, W]
     * @return Augmented tensor
     */
    private Tensor augment(Tensor tensor) {
        // Random 90-degree rotations around z-axis
        if (random.nextDouble() < 0.5) {
            tensor = rotate90Z(tensor);
        }

        // Random flips
        if (random.nextDouble() < 0.5) {
            tensor = flipX(tensor);
        }
        if (random.nextDouble() < 0.5) {
            tensor = flipY(tensor);
        }
        if (random.nextDouble() < 0.5) {
            tensor = flipZ(tensor);
        }

        // Gaussian noise (10% of time)
        if (random.nextDouble() < 0.1) {
            tensor = addGaussianNoise(tensor, 0.01);
        }

        // Brightness/contrast adjustment (20% of time)
        if (random.nextDouble() < 0.2) {
            double factor = 0.9 + random.nextDouble() * 0.2; // [0.9, 1.1]
            tensor = tensor.mul(factor);
        }

        return tensor;
    }

    /**
     * Rotates tensor 90 degrees around z-axis.
     */
    private Tensor rotate90Z(Tensor tensor) {
        // Rotate in XY plane (dimensions 3 and 4)
        return tensor.rot90(1, new long[]{3, 4});
    }

    /**
     * Flips tensor along X dimension.
     */
    private Tensor flipX(Tensor tensor) {
        return tensor.flip(new long[]{4}); // Width dimension
    }

    /**
     * Flips tensor along Y dimension.
     */
    private Tensor flipY(Tensor tensor) {
        return tensor.flip(new long[]{3}); // Height dimension
    }

    /**
     * Flips tensor along Z dimension.
     */
    private Tensor flipZ(Tensor tensor) {
        return tensor.flip(new long[]{2}); // Depth dimension
    }

    /**
     * Adds Gaussian noise to tensor.
     */
    private Tensor addGaussianNoise(Tensor tensor, double stddev) {
        Tensor noise = org.bytedeco.pytorch.global.torch.randn_like(tensor);
        noise = noise.mul(stddev);
        return tensor.add(noise);
    }

    /**
     * Stacks list of tensors into a batch tensor.
     *
     * @param tensors List of tensors [1, C, D, H, W]
     * @return Batch tensor [B, C, D, H, W]
     */
    private Tensor stackTensors(List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalStateException("No valid tensors in batch");
        }

        // Convert list to array
        Tensor[] tensorArray = tensors.toArray(new Tensor[0]);

        // Stack along batch dimension
        return org.bytedeco.pytorch.global.torch.cat(tensorArray, 0);
    }

    /**
     * Container for a batch of data.
     */
    public static class Batch {
        private final List<MicroObject> cells;
        private final int index;
        private Tensor data;

        public Batch(List<MicroObject> cells, int index) {
            this.cells = cells;
            this.index = index;
        }

        public List<MicroObject> getCells() {
            return cells;
        }

        public int getIndex() {
            return index;
        }

        public Tensor getData() {
            return data;
        }

        void setData(Tensor data) {
            this.data = data;
        }

        public int getSize() {
            return cells.size();
        }
    }
}
