package vtea.deeplearning.data;

import ij.ImageStack;
import ij.process.ImageProcessor;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Utility class for converting between ImageJ ImageStack and PyTorch Tensor.
 *
 * <p>This class handles the conversion of 3D volumetric image data between
 * ImageJ's ImageStack format and PyTorch's Tensor format, which is required
 * for deep learning operations.</p>
 *
 * <p>Supports:</p>
 * <ul>
 *   <li>Single and multi-channel conversion</li>
 *   <li>Multiple normalization strategies (Z-score, Min-Max, None)</li>
 *   <li>Batch tensor creation</li>
 *   <li>Bidirectional conversion (ImageStack ↔ Tensor)</li>
 * </ul>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class TensorConverter {

    private static final Logger logger = LoggerFactory.getLogger(TensorConverter.class);

    private final NormalizationType normalization;
    private final boolean useGPU;

    /**
     * Normalization strategies for tensor data.
     */
    public enum NormalizationType {
        /** Z-score normalization: (x - mean) / std */
        ZSCORE,

        /** Min-max normalization to [0, 1] range */
        MINMAX,

        /** No normalization */
        NONE
    }

    /**
     * Creates a TensorConverter with specified normalization and device settings.
     *
     * @param normalization The normalization strategy to use
     * @param useGPU Whether to create tensors on GPU (requires CUDA)
     */
    public TensorConverter(NormalizationType normalization, boolean useGPU) {
        this.normalization = normalization;
        this.useGPU = useGPU && cuda_is_available();

        if (useGPU && !cuda_is_available()) {
            logger.warn("GPU requested but CUDA not available. Falling back to CPU.");
        }

        logger.info("TensorConverter initialized: normalization={}, device={}",
                   normalization, this.useGPU ? "CUDA" : "CPU");
    }

    /**
     * Creates a TensorConverter with default settings (ZSCORE normalization, CPU).
     */
    public TensorConverter() {
        this(NormalizationType.ZSCORE, false);
    }

    /**
     * Converts a single-channel ImageStack to a PyTorch Tensor.
     *
     * @param stack The ImageStack to convert (depth × height × width)
     * @return Tensor with shape [1, 1, depth, height, width] (batch size 1, 1 channel)
     */
    public Tensor imageStackToTensor(ImageStack stack) {
        if (stack == null || stack.getSize() == 0) {
            throw new IllegalArgumentException("ImageStack is null or empty");
        }

        int depth = stack.getSize();
        int height = stack.getHeight();
        int width = stack.getWidth();

        logger.debug("Converting ImageStack to Tensor: {}×{}×{}", depth, height, width);

        // Extract data from ImageStack
        float[] data = extractDataFromStack(stack);

        // Normalize
        data = normalize(data);

        // Create tensor
        long[] shape = new long[]{1, 1, depth, height, width}; // [B, C, D, H, W]
        Tensor tensor = createTensorFromArray(data, shape);

        return tensor;
    }

    /**
     * Converts multi-channel ImageStacks to a PyTorch Tensor.
     *
     * @param stacks Array of ImageStacks, one per channel
     * @return Tensor with shape [1, C, depth, height, width]
     */
    public Tensor imageStacksToTensor(ImageStack[] stacks) {
        if (stacks == null || stacks.length == 0) {
            throw new IllegalArgumentException("ImageStack array is null or empty");
        }

        int numChannels = stacks.length;
        int depth = stacks[0].getSize();
        int height = stacks[0].getHeight();
        int width = stacks[0].getWidth();

        // Validate all stacks have same dimensions
        for (int i = 1; i < numChannels; i++) {
            if (stacks[i].getSize() != depth ||
                stacks[i].getHeight() != height ||
                stacks[i].getWidth() != width) {
                throw new IllegalArgumentException(
                    String.format("Channel %d has inconsistent dimensions", i));
            }
        }

        logger.debug("Converting {} channel ImageStacks to Tensor: {}×{}×{}",
                    numChannels, depth, height, width);

        // Extract and concatenate all channels
        int volumeSize = depth * height * width;
        float[] data = new float[numChannels * volumeSize];

        for (int c = 0; c < numChannels; c++) {
            float[] channelData = extractDataFromStack(stacks[c]);
            channelData = normalize(channelData);
            System.arraycopy(channelData, 0, data, c * volumeSize, volumeSize);
        }

        // Create tensor
        long[] shape = new long[]{1, numChannels, depth, height, width};
        Tensor tensor = createTensorFromArray(data, shape);

        return tensor;
    }

    /**
     * Converts a PyTorch Tensor back to ImageStack.
     *
     * @param tensor Tensor with shape [1, 1, D, H, W] or [1, C, D, H, W]
     * @return ImageStack (if single channel) or array of ImageStacks (if multi-channel)
     */
    public ImageStack tensorToImageStack(Tensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("Tensor is null");
        }

        // Validate tensor dimensions
        long[] shape = tensor.sizes();
        if (shape.length != 5) {
            throw new IllegalArgumentException(
                "Tensor must be 5D [B, C, D, H, W], got shape: " +
                java.util.Arrays.toString(shape));
        }

        int batch = (int) shape[0];
        int channels = (int) shape[1];
        int depth = (int) shape[2];
        int height = (int) shape[3];
        int width = (int) shape[4];

        if (batch != 1) {
            throw new IllegalArgumentException("Batch size must be 1, got: " + batch);
        }

        if (channels != 1) {
            throw new UnsupportedOperationException(
                "Multi-channel tensor to ImageStack conversion not yet implemented. " +
                "Use tensorToImageStacks() instead.");
        }

        logger.debug("Converting Tensor to ImageStack: {}×{}×{}", depth, height, width);

        // Move to CPU if on GPU
        Tensor cpuTensor = tensor.cpu();

        // Extract data
        float[] data = tensorToFloatArray(cpuTensor);

        // Create ImageStack
        ImageStack stack = new ImageStack(width, height);
        int sliceSize = height * width;

        for (int z = 0; z < depth; z++) {
            float[] sliceData = new float[sliceSize];
            System.arraycopy(data, z * sliceSize, sliceData, 0, sliceSize);

            ij.process.FloatProcessor fp = new ij.process.FloatProcessor(width, height, sliceData);
            stack.addSlice("z=" + (z + 1), fp);
        }

        return stack;
    }

    /**
     * Creates a batch tensor from multiple ImageStacks.
     *
     * @param stacks Array of ImageStacks
     * @return Tensor with shape [N, 1, D, H, W] where N is batch size
     */
    public Tensor createBatch(ImageStack[] stacks) {
        if (stacks == null || stacks.length == 0) {
            throw new IllegalArgumentException("Stacks array is null or empty");
        }

        int batchSize = stacks.length;

        // Convert each stack individually
        Tensor[] tensors = new Tensor[batchSize];
        for (int i = 0; i < batchSize; i++) {
            tensors[i] = imageStackToTensor(stacks[i]);
        }

        // Concatenate along batch dimension
        return cat(tensors, 0);
    }

    // ==================== Private Helper Methods ====================

    /**
     * Extracts pixel data from ImageStack as float array.
     */
    private float[] extractDataFromStack(ImageStack stack) {
        int depth = stack.getSize();
        int height = stack.getHeight();
        int width = stack.getWidth();
        int volumeSize = depth * height * width;

        float[] data = new float[volumeSize];
        int index = 0;

        for (int z = 1; z <= depth; z++) { // ImageStack is 1-indexed
            ImageProcessor ip = stack.getProcessor(z);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    data[index++] = ip.getf(x, y);
                }
            }
        }

        return data;
    }

    /**
     * Normalizes data according to the normalization strategy.
     */
    private float[] normalize(float[] data) {
        switch (normalization) {
            case ZSCORE:
                return normalizeZScore(data);
            case MINMAX:
                return normalizeMinMax(data);
            case NONE:
                return data;
            default:
                throw new IllegalStateException("Unknown normalization: " + normalization);
        }
    }

    /**
     * Z-score normalization: (x - mean) / std
     */
    private float[] normalizeZScore(float[] data) {
        // Compute mean
        double sum = 0.0;
        for (float value : data) {
            sum += value;
        }
        double mean = sum / data.length;

        // Compute standard deviation
        double sumSquaredDiff = 0.0;
        for (float value : data) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        double std = Math.sqrt(sumSquaredDiff / data.length);

        // Avoid division by zero
        if (std < 1e-8) {
            logger.warn("Standard deviation near zero ({}), skipping normalization", std);
            return data;
        }

        // Normalize
        float[] normalized = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            normalized[i] = (float) ((data[i] - mean) / std);
        }

        logger.debug("Z-score normalization: mean={}, std={}", mean, std);
        return normalized;
    }

    /**
     * Min-max normalization to [0, 1] range
     */
    private float[] normalizeMinMax(float[] data) {
        // Find min and max
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;

        for (float value : data) {
            if (value < min) min = value;
            if (value > max) max = value;
        }

        float range = max - min;

        // Avoid division by zero
        if (range < 1e-8) {
            logger.warn("Range near zero ({}), skipping normalization", range);
            return data;
        }

        // Normalize
        float[] normalized = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            normalized[i] = (data[i] - min) / range;
        }

        logger.debug("Min-max normalization: min={}, max={}", min, max);
        return normalized;
    }

    /**
     * Creates a PyTorch tensor from float array with given shape.
     */
    private Tensor createTensorFromArray(float[] data, long[] shape) {
        // Create tensor options
        TensorOptions options = new TensorOptions();
        options.dtype(kFloat32);

        if (useGPU) {
            options.device(kCUDA);
        }

        // Create float pointer
        FloatPointer pointer = new FloatPointer(data);

        // Create shape pointer
        LongPointer shapePointer = new LongPointer(shape);

        // Create tensor
        Tensor tensor = from_blob(pointer, shapePointer, options);

        // Clone to ensure data ownership
        tensor = tensor.clone();

        return tensor;
    }

    /**
     * Converts tensor to float array.
     */
    private float[] tensorToFloatArray(Tensor tensor) {
        long numElements = 1;
        long[] shape = tensor.sizes();
        for (long dim : shape) {
            numElements *= dim;
        }

        FloatPointer pointer = new FloatPointer(numElements);
        tensor.data_ptr_float(pointer);

        float[] data = new float[(int) numElements];
        pointer.get(data);

        return data;
    }

    /**
     * Checks if CUDA is available for GPU acceleration.
     *
     * @return true if CUDA is available
     */
    public static boolean isCudaAvailable() {
        try {
            return cuda_is_available();
        } catch (Exception e) {
            logger.warn("Error checking CUDA availability", e);
            return false;
        }
    }
}
