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

/**
 * NephNet3D - 3D Convolutional Neural Network for kidney cell classification.
 *
 * Architecture based on: https://github.com/awoloshuk/NephNet
 * Reference: Woloshuk et al., "RIGI: A novel 3D imaging informatics pipeline for kidney histopathology"
 *
 * Architecture:
 * - 4 Convolutional blocks with progressive channel expansion (32→64→128→256)
 * - Each block: Conv3d → BatchNorm3d → LeakyReLU → MaxPool3d
 * - Fully connected classifier: 256 → 256 → 128 → num_classes
 * - Kaiming initialization for weights
 *
 * Input: [batch, channels, depth, height, width]
 * Output: [batch, num_classes]
 *
 * @author VTEA Deep Learning Team
 */
public class NephNet3D extends AbstractDeepLearningModel {

    // Convolutional blocks
    private transient SequentialImpl conv1;
    private transient SequentialImpl conv2;
    private transient SequentialImpl conv3;
    private transient SequentialImpl conv4;

    // Classifier (fully connected layers)
    private transient SequentialImpl classifier;

    // Pooling layer
    private transient AdaptiveAvgPool3dImpl globalPool;

    // Architecture hyperparameters
    private final int[] channels = {32, 64, 128, 256};  // Channel progression
    private final double leakyReluSlope = 0.01;         // LeakyReLU negative slope

    /**
     * Constructor with default input shape [64, 64, 64]
     */
    public NephNet3D(int inputChannels, int numClasses) {
        this(inputChannels, numClasses, new int[]{64, 64, 64});
    }

    /**
     * Constructor with custom input shape
     */
    public NephNet3D(int inputChannels, int numClasses, int[] inputShape) {
        super("NephNet3D", inputChannels, numClasses, inputShape);
    }

    /**
     * Build the NephNet3D architecture
     */
    @Override
    protected void buildArchitecture() {
        // Create the main network module
        network = new SequentialImpl();

        // Convolutional Block 1: input_channels → 32
        conv1 = new SequentialImpl();
        conv1.push_back(new Conv3dImpl(
            new Conv3dOptions(inputChannels, channels[0], new long[]{3, 3, 3})
                .padding(new long[]{1, 1, 1})));
        conv1.push_back(new BatchNorm3dImpl(channels[0]));
        conv1.push_back(new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(leakyReluSlope)));
        conv1.push_back(new MaxPool3dImpl(
            new MaxPool3dOptions(new long[]{2, 2, 2})
                .stride(new long[]{2, 2, 2})));

        // Convolutional Block 2: 32 → 64
        conv2 = new SequentialImpl();
        conv2.push_back(new Conv3dImpl(
            new Conv3dOptions(channels[0], channels[1], new long[]{3, 3, 3})
                .padding(new long[]{1, 1, 1})));
        conv2.push_back(new BatchNorm3dImpl(channels[1]));
        conv2.push_back(new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(leakyReluSlope)));
        conv2.push_back(new MaxPool3dImpl(
            new MaxPool3dOptions(new long[]{2, 2, 2})
                .stride(new long[]{2, 2, 2})));

        // Convolutional Block 3: 64 → 128
        conv3 = new SequentialImpl();
        conv3.push_back(new Conv3dImpl(
            new Conv3dOptions(channels[1], channels[2], new long[]{3, 3, 3})
                .padding(new long[]{1, 1, 1})));
        conv3.push_back(new BatchNorm3dImpl(channels[2]));
        conv3.push_back(new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(leakyReluSlope)));
        conv3.push_back(new MaxPool3dImpl(
            new MaxPool3dOptions(new long[]{2, 2, 2})
                .stride(new long[]{2, 2, 2})));

        // Convolutional Block 4: 128 → 256
        conv4 = new SequentialImpl();
        conv4.push_back(new Conv3dImpl(
            new Conv3dOptions(channels[2], channels[3], new long[]{3, 3, 3})
                .padding(new long[]{1, 1, 1})));
        conv4.push_back(new BatchNorm3dImpl(channels[3]));
        conv4.push_back(new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(leakyReluSlope)));
        conv4.push_back(new MaxPool3dImpl(
            new MaxPool3dOptions(new long[]{2, 2, 2})
                .stride(new long[]{2, 2, 2})));

        // Global Average Pooling
        globalPool = new AdaptiveAvgPool3dImpl(new long[]{1, 1, 1});

        // Fully Connected Classifier: 256 → 256 → 128 → num_classes
        classifier = new SequentialImpl();
        classifier.push_back(new LinearImpl(channels[3], 256));
        classifier.push_back(new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(leakyReluSlope)));
        classifier.push_back(new DropoutImpl(0.5));  // Dropout for regularization

        classifier.push_back(new LinearImpl(256, 128));
        classifier.push_back(new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(leakyReluSlope)));
        classifier.push_back(new DropoutImpl(0.5));

        classifier.push_back(new LinearImpl(128, numClasses));

        // Assemble the complete network
        ((SequentialImpl) network).push_back(conv1);
        ((SequentialImpl) network).push_back(conv2);
        ((SequentialImpl) network).push_back(conv3);
        ((SequentialImpl) network).push_back(conv4);

        System.out.println("NephNet3D architecture built successfully");
    }

    /**
     * Forward pass through the network
     *
     * @param input Input tensor [batch, channels, depth, height, width]
     * @return Output logits [batch, num_classes]
     */
    @Override
    public Tensor forward(Tensor input) {
        if (!isBuilt) {
            build();
        }

        // Validate input shape
        long[] inputSizes = new long[(int) input.dim()];
        for (int i = 0; i < inputSizes.length; i++) {
            inputSizes[i] = input.size(i);
        }

        if (inputSizes.length != 5) {
            throw new IllegalArgumentException("Input must have 5 dimensions [batch, channels, depth, height, width]");
        }

        if (inputSizes[1] != inputChannels) {
            throw new IllegalArgumentException("Input channels mismatch: expected " + inputChannels +
                                             ", got " + inputSizes[1]);
        }

        // Pass through convolutional blocks
        Tensor x = input;
        x = conv1.forward(x);
        x = conv2.forward(x);
        x = conv3.forward(x);
        x = conv4.forward(x);

        // Global average pooling
        x = globalPool.forward(x);

        // Flatten for fully connected layers
        x = x.view(new long[]{x.size(0), -1});

        // Pass through classifier
        x = classifier.forward(x);

        return x;
    }

    /**
     * Initialize weights using Kaiming (He) initialization
     * This is appropriate for LeakyReLU activations
     */
    @Override
    protected void initializeWeights() {
        if (network == null) {
            throw new IllegalStateException("Network must be built before initializing weights");
        }

        // Initialize convolutional layers
        initializeBlockWeights(conv1);
        initializeBlockWeights(conv2);
        initializeBlockWeights(conv3);
        initializeBlockWeights(conv4);

        // Initialize classifier layers
        initializeBlockWeights(classifier);

        System.out.println("NephNet3D weights initialized with Kaiming initialization");
    }

    /**
     * Initialize weights for a sequential block
     */
    private void initializeBlockWeights(SequentialImpl block) {
        NamedModuleIterator modules = block.named_modules();

        while (modules.hasNext()) {
            NamedModule namedModule = modules.next();
            Module module = namedModule.value();
            String name = namedModule.key().getString();

            // Initialize Conv3d layers
            if (module instanceof Conv3dImpl) {
                Conv3dImpl conv = (Conv3dImpl) module;
                Tensor weight = conv.weight();

                // Kaiming initialization for convolutional layers
                // fan_mode = "fan_in", nonlinearity = "leaky_relu"
                kaiming_normal_(weight, leakyReluSlope, FanModeType.FAN_IN, NonlinearityType.LeakyReLU);

                // Initialize bias to zero if present
                if (conv.bias() != null && !conv.bias().isNull()) {
                    zeros_(conv.bias());
                }
            }

            // Initialize Linear layers
            if (module instanceof LinearImpl) {
                LinearImpl linear = (LinearImpl) module;
                Tensor weight = linear.weight();

                // Kaiming initialization for linear layers
                kaiming_normal_(weight, leakyReluSlope, FanModeType.FAN_IN, NonlinearityType.LeakyReLU);

                // Initialize bias to zero if present
                if (linear.bias() != null && !linear.bias().isNull()) {
                    zeros_(linear.bias());
                }
            }

            // BatchNorm layers are initialized automatically
        }
    }

    /**
     * Get architecture description
     */
    public String getArchitectureDescription() {
        StringBuilder sb = new StringBuilder();
        sb.append("NephNet3D Architecture:\n");
        sb.append("======================\n");
        sb.append(String.format("Input: [batch, %d, %d, %d, %d]\n",
                               inputChannels, inputShape[0], inputShape[1], inputShape[2]));
        sb.append("\nConvolutional Blocks:\n");
        sb.append(String.format("  Block 1: %d → %d channels (Conv3d 3×3×3 → BN → LeakyReLU → MaxPool 2×2×2)\n",
                               inputChannels, channels[0]));
        sb.append(String.format("  Block 2: %d → %d channels (Conv3d 3×3×3 → BN → LeakyReLU → MaxPool 2×2×2)\n",
                               channels[0], channels[1]));
        sb.append(String.format("  Block 3: %d → %d channels (Conv3d 3×3×3 → BN → LeakyReLU → MaxPool 2×2×2)\n",
                               channels[1], channels[2]));
        sb.append(String.format("  Block 4: %d → %d channels (Conv3d 3×3×3 → BN → LeakyReLU → MaxPool 2×2×2)\n",
                               channels[2], channels[3]));
        sb.append("\nClassifier:\n");
        sb.append("  Global Average Pooling (1×1×1)\n");
        sb.append(String.format("  FC: %d → 256 → LeakyReLU → Dropout(0.5)\n", channels[3]));
        sb.append("  FC: 256 → 128 → LeakyReLU → Dropout(0.5)\n");
        sb.append(String.format("  FC: 128 → %d (output logits)\n", numClasses));
        sb.append(String.format("\nOutput: [batch, %d]\n", numClasses));

        if (isBuilt) {
            sb.append(String.format("\nTotal Parameters: %,d\n", getParameterCount()));
            sb.append(String.format("Memory Footprint: %.2f MB\n", getMemoryFootprintMB()));
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("NephNet3D[input=%dx%dx%dx%d, classes=%d, params=%s]",
                           inputChannels, inputShape[0], inputShape[1], inputShape[2],
                           numClasses, isBuilt ? String.format("%,d", getParameterCount()) : "not built");
    }

    /**
     * Factory method for binary classification (2 classes)
     */
    public static NephNet3D forBinaryClassification(int inputChannels, int[] inputShape) {
        return new NephNet3D(inputChannels, 2, inputShape);
    }

    /**
     * Factory method for multi-class classification
     */
    public static NephNet3D forMultiClassClassification(int inputChannels, int numClasses, int[] inputShape) {
        if (numClasses < 2) {
            throw new IllegalArgumentException("Number of classes must be at least 2");
        }
        return new NephNet3D(inputChannels, numClasses, inputShape);
    }

    /**
     * Factory method with default cubic region size
     */
    public static NephNet3D withCubicRegion(int inputChannels, int numClasses, int regionSize) {
        return new NephNet3D(inputChannels, numClasses, new int[]{regionSize, regionSize, regionSize});
    }
}
