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
package vtea.deeplearning.data;

import ij.ImageStack;
import ij.process.ImageProcessor;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.ArrayList;

/**
 * Utility class for converting between ImageJ ImageStack and PyTorch Tensor formats.
 * Handles multi-channel 3D images and various normalization strategies.
 *
 * @author VTEA Deep Learning Team
 */
public class TensorConverter {

    /**
     * Normalization types for tensor conversion
     */
    public enum NormalizationType {
        NONE,       // No normalization
        MINMAX,     // Min-max normalization to [0, 1]
        ZSCORE,     // Z-score normalization (mean=0, std=1)
        PERCENTILE  // Percentile-based normalization (robust to outliers)
    }

    /**
     * Convert a single-channel ImageStack to PyTorch Tensor.
     * Output shape: [1, depth, height, width] (batch size 1, single channel)
     *
     * @param stack       ImageStack to convert
     * @param normType    Type of normalization to apply
     * @return PyTorch Tensor
     */
    public static Tensor imageStackToTensor(ImageStack stack, NormalizationType normType) {
        if (stack == null) {
            throw new IllegalArgumentException("ImageStack cannot be null");
        }

        int width = stack.getWidth();
        int height = stack.getHeight();
        int depth = stack.getSize();

        // Create float array to hold pixel data
        float[] data = new float[depth * height * width];

        // Extract pixels from stack
        int index = 0;
        for (int z = 0; z < depth; z++) {
            ImageProcessor ip = stack.getProcessor(z + 1); // ImageJ uses 1-based indexing
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    data[index++] = ip.getPixelValue(x, y);
                }
            }
        }

        // Apply normalization
        data = normalize(data, normType);

        // Create tensor from data
        // Shape: [1, 1, depth, height, width] (batch=1, channels=1)
        FloatBuffer buffer = FloatBuffer.wrap(data);
        LongVector shape = new LongVector(1, 1, depth, height, width);
        Tensor tensor = from_blob(buffer, shape);

        return tensor;
    }

    /**
     * Convert multi-channel ImageStack array to PyTorch Tensor.
     * Output shape: [1, channels, depth, height, width]
     *
     * @param stacks       Array of ImageStacks (one per channel)
     * @param channelIndices Which channels to include (null = all)
     * @param normType     Type of normalization to apply
     * @return PyTorch Tensor
     */
    public static Tensor multiChannelToTensor(ImageStack[] stacks, int[] channelIndices,
                                               NormalizationType normType) {
        if (stacks == null || stacks.length == 0) {
            throw new IllegalArgumentException("ImageStack array cannot be null or empty");
        }

        // Determine which channels to use
        int[] channels = channelIndices;
        if (channels == null) {
            channels = new int[stacks.length];
            for (int i = 0; i < channels.length; i++) {
                channels[i] = i;
            }
        }

        int numChannels = channels.length;
        int width = stacks[0].getWidth();
        int height = stacks[0].getHeight();
        int depth = stacks[0].getSize();

        // Validate all stacks have same dimensions
        for (int ch : channels) {
            if (stacks[ch].getWidth() != width ||
                stacks[ch].getHeight() != height ||
                stacks[ch].getSize() != depth) {
                throw new IllegalArgumentException("All ImageStacks must have the same dimensions");
            }
        }

        // Create float array for all channels
        float[] data = new float[numChannels * depth * height * width];

        // Extract pixels from all channels
        for (int c = 0; c < numChannels; c++) {
            ImageStack stack = stacks[channels[c]];
            int channelOffset = c * depth * height * width;

            for (int z = 0; z < depth; z++) {
                ImageProcessor ip = stack.getProcessor(z + 1);
                int sliceOffset = z * height * width;

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int index = channelOffset + sliceOffset + y * width + x;
                        data[index] = ip.getPixelValue(x, y);
                    }
                }
            }
        }

        // Apply normalization
        data = normalize(data, normType);

        // Create tensor
        // Shape: [1, numChannels, depth, height, width]
        FloatBuffer buffer = FloatBuffer.wrap(data);
        LongVector shape = new LongVector(1, numChannels, depth, height, width);
        Tensor tensor = from_blob(buffer, shape);

        return tensor;
    }

    /**
     * Convert a list of multi-channel regions to a batched tensor.
     * Output shape: [batchSize, channels, depth, height, width]
     *
     * @param regions      List of ImageStack arrays (each array is one region with multiple channels)
     * @param channelIndices Which channels to include
     * @param normType     Type of normalization
     * @return PyTorch Tensor
     */
    public static Tensor batchRegionsToTensor(List<ImageStack[]> regions, int[] channelIndices,
                                               NormalizationType normType) {
        if (regions == null || regions.isEmpty()) {
            throw new IllegalArgumentException("Regions list cannot be null or empty");
        }

        int batchSize = regions.size();
        List<Tensor> tensors = new ArrayList<>();

        // Convert each region to tensor
        for (ImageStack[] region : regions) {
            Tensor t = multiChannelToTensor(region, channelIndices, normType);
            tensors.add(t.squeeze(0)); // Remove batch dimension for concatenation
        }

        // Stack tensors along batch dimension
        TensorVector tensorVec = new TensorVector(tensors.size());
        for (int i = 0; i < tensors.size(); i++) {
            tensorVec.put(i, tensors.get(i));
        }

        Tensor batchTensor = stack(tensorVec, 0); // Stack along dimension 0

        return batchTensor;
    }

    /**
     * Convert PyTorch Tensor back to ImageStack.
     * Input tensor should have shape [depth, height, width] or [1, depth, height, width]
     *
     * @param tensor PyTorch Tensor to convert
     * @return ImageStack
     */
    public static ImageStack tensorToImageStack(Tensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("Tensor cannot be null");
        }

        // Get tensor dimensions
        LongVector sizes = tensor.sizes();
        int ndim = (int) sizes.size();

        int depth, height, width;
        if (ndim == 3) {
            depth = (int) sizes.get(0);
            height = (int) sizes.get(1);
            width = (int) sizes.get(2);
        } else if (ndim == 4) {
            // Assume shape [1, depth, height, width]
            depth = (int) sizes.get(1);
            height = (int) sizes.get(2);
            width = (int) sizes.get(3);
        } else {
            throw new IllegalArgumentException("Tensor must have 3 or 4 dimensions");
        }

        // Create ImageStack
        ImageStack stack = new ImageStack(width, height, depth);

        // Extract data from tensor
        FloatPointer dataPtr = tensor.data_ptr_float();
        float[] data = new float[depth * height * width];
        dataPtr.get(data);

        // Populate ImageStack
        for (int z = 0; z < depth; z++) {
            float[] sliceData = new float[height * width];
            System.arraycopy(data, z * height * width, sliceData, 0, height * width);

            ij.process.FloatProcessor fp = new ij.process.FloatProcessor(width, height, sliceData);
            stack.setProcessor(fp, z + 1);
        }

        return stack;
    }

    /**
     * Normalize data array based on normalization type
     *
     * @param data     Data to normalize
     * @param normType Type of normalization
     * @return Normalized data
     */
    private static float[] normalize(float[] data, NormalizationType normType) {
        if (normType == NormalizationType.NONE) {
            return data;
        }

        float[] normalized = new float[data.length];

        switch (normType) {
            case MINMAX:
                normalized = normalizeMinMax(data);
                break;
            case ZSCORE:
                normalized = normalizeZScore(data);
                break;
            case PERCENTILE:
                normalized = normalizePercentile(data, 1.0f, 99.0f);
                break;
            default:
                System.arraycopy(data, 0, normalized, 0, data.length);
        }

        return normalized;
    }

    /**
     * Min-max normalization to [0, 1] range
     */
    private static float[] normalizeMinMax(float[] data) {
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;

        // Find min and max
        for (float value : data) {
            if (value < min) min = value;
            if (value > max) max = value;
        }

        // Normalize
        float[] normalized = new float[data.length];
        float range = max - min;
        if (range > 0) {
            for (int i = 0; i < data.length; i++) {
                normalized[i] = (data[i] - min) / range;
            }
        } else {
            // All values are the same
            for (int i = 0; i < data.length; i++) {
                normalized[i] = 0.5f;
            }
        }

        return normalized;
    }

    /**
     * Z-score normalization (mean=0, std=1)
     */
    private static float[] normalizeZScore(float[] data) {
        // Calculate mean
        double sum = 0;
        for (float value : data) {
            sum += value;
        }
        double mean = sum / data.length;

        // Calculate standard deviation
        double variance = 0;
        for (float value : data) {
            variance += (value - mean) * (value - mean);
        }
        double std = Math.sqrt(variance / data.length);

        // Normalize
        float[] normalized = new float[data.length];
        if (std > 0) {
            for (int i = 0; i < data.length; i++) {
                normalized[i] = (float) ((data[i] - mean) / std);
            }
        } else {
            // All values are the same
            for (int i = 0; i < data.length; i++) {
                normalized[i] = 0.0f;
            }
        }

        return normalized;
    }

    /**
     * Percentile-based normalization (robust to outliers)
     */
    private static float[] normalizePercentile(float[] data, float lowPerc, float highPerc) {
        // Sort data for percentile calculation
        float[] sorted = data.clone();
        java.util.Arrays.sort(sorted);

        int lowIdx = (int) (sorted.length * lowPerc / 100.0);
        int highIdx = (int) (sorted.length * highPerc / 100.0);

        float lowVal = sorted[lowIdx];
        float highVal = sorted[highIdx];

        // Normalize
        float[] normalized = new float[data.length];
        float range = highVal - lowVal;
        if (range > 0) {
            for (int i = 0; i < data.length; i++) {
                float val = (data[i] - lowVal) / range;
                // Clip to [0, 1]
                normalized[i] = Math.max(0.0f, Math.min(1.0f, val));
            }
        } else {
            for (int i = 0; i < data.length; i++) {
                normalized[i] = 0.5f;
            }
        }

        return normalized;
    }

    /**
     * Normalize a single tensor in-place using Z-score normalization
     */
    public static Tensor normalizeZScore(Tensor tensor) {
        Tensor mean = tensor.mean();
        Tensor std = tensor.std();
        return tensor.sub(mean).div(std.add(1e-8)); // Add small epsilon to avoid division by zero
    }

    /**
     * Normalize a single tensor in-place using min-max normalization
     */
    public static Tensor normalizeMinMax(Tensor tensor) {
        Tensor min = tensor.min();
        Tensor max = tensor.max();
        Tensor range = max.sub(min);
        return tensor.sub(min).div(range.add(1e-8));
    }
}
