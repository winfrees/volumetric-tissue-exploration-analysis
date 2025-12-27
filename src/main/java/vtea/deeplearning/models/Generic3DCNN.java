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

import java.util.ArrayList;
import java.util.List;

/**
 * Generic configurable 3D Convolutional Neural Network.
 * Provides flexible architecture through builder pattern.
 *
 * Features:
 * - Configurable number of convolutional blocks
 * - Customizable channel progression
 * - Choice of activation functions (ReLU, LeakyReLU, ELU)
 * - Optional batch normalization and dropout
 * - Flexible pooling strategies
 * - Custom classifier architecture
 *
 * @author VTEA Deep Learning Team
 */
public class Generic3DCNN extends AbstractDeepLearningModel {

    // Architecture configuration
    private final int[] blockChannels;      // Channels for each conv block
    private final int[] kernelSizes;        // Kernel size for each block (cubic)
    private final int[] poolingSizes;       // Pooling size for each block
    private final ActivationType activation;
    private final boolean useBatchNorm;
    private final double dropoutRate;
    private final int[] fcLayers;           // Hidden layer sizes in classifier

    // Network components
    private transient List<SequentialImpl> convBlocks;
    private transient AdaptiveAvgPool3dImpl globalPool;
    private transient SequentialImpl classifier;

    /**
     * Activation function types
     */
    public enum ActivationType {
        RELU,
        LEAKY_RELU,
        ELU
    }

    /**
     * Private constructor - use Builder
     */
    private Generic3DCNN(Builder builder) {
        super(builder.modelName, builder.inputChannels, builder.numClasses, builder.inputShape);

        this.blockChannels = builder.blockChannels;
        this.kernelSizes = builder.kernelSizes;
        this.poolingSizes = builder.poolingSizes;
        this.activation = builder.activation;
        this.useBatchNorm = builder.useBatchNorm;
        this.dropoutRate = builder.dropoutRate;
        this.fcLayers = builder.fcLayers;
    }

    @Override
    protected void buildArchitecture() {
        network = new SequentialImpl();
        convBlocks = new ArrayList<>();

        // Build convolutional blocks
        int inChannels = inputChannels;
        for (int i = 0; i < blockChannels.length; i++) {
            int outChannels = blockChannels[i];
            int kernelSize = kernelSizes[i];
            int poolSize = poolingSizes[i];

            SequentialImpl block = buildConvBlock(inChannels, outChannels, kernelSize, poolSize);
            convBlocks.add(block);
            ((SequentialImpl) network).push_back(block);

            inChannels = outChannels;
        }

        // Global average pooling
        globalPool = new AdaptiveAvgPool3dImpl(new long[]{1, 1, 1});

        // Build classifier
        classifier = buildClassifier(blockChannels[blockChannels.length - 1]);

        System.out.println("Generic3DCNN architecture built successfully");
    }

    /**
     * Build a single convolutional block
     */
    private SequentialImpl buildConvBlock(int inChannels, int outChannels, int kernelSize, int poolSize) {
        SequentialImpl block = new SequentialImpl();

        // Convolutional layer
        int padding = kernelSize / 2;  // Same padding
        block.push_back(new Conv3dImpl(
            new Conv3dOptions(inChannels, outChannels, new long[]{kernelSize, kernelSize, kernelSize})
                .padding(new long[]{padding, padding, padding})));

        // Batch normalization (optional)
        if (useBatchNorm) {
            block.push_back(new BatchNorm3dImpl(outChannels));
        }

        // Activation function
        block.push_back(createActivation());

        // Pooling layer
        if (poolSize > 1) {
            block.push_back(new MaxPool3dImpl(
                new MaxPool3dOptions(new long[]{poolSize, poolSize, poolSize})
                    .stride(new long[]{poolSize, poolSize, poolSize})));
        }

        return block;
    }

    /**
     * Build the classifier (fully connected layers)
     */
    private SequentialImpl buildClassifier(int inputFeatures) {
        SequentialImpl fc = new SequentialImpl();

        int inFeatures = inputFeatures;
        for (int hiddenSize : fcLayers) {
            fc.push_back(new LinearImpl(inFeatures, hiddenSize));
            fc.push_back(createActivation());

            if (dropoutRate > 0) {
                fc.push_back(new DropoutImpl(dropoutRate));
            }

            inFeatures = hiddenSize;
        }

        // Output layer
        fc.push_back(new LinearImpl(inFeatures, numClasses));

        return fc;
    }

    /**
     * Create activation module based on configuration
     */
    private Module createActivation() {
        switch (activation) {
            case RELU:
                return new ReLUImpl();
            case LEAKY_RELU:
                return new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(0.01));
            case ELU:
                return new ELUImpl();
            default:
                return new ReLUImpl();
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        if (!isBuilt) {
            build();
        }

        // Validate input
        if (input.dim() != 5) {
            throw new IllegalArgumentException("Input must have 5 dimensions [batch, channels, depth, height, width]");
        }
        if (input.size(1) != inputChannels) {
            throw new IllegalArgumentException("Input channels mismatch: expected " + inputChannels +
                                             ", got " + input.size(1));
        }

        // Forward through conv blocks
        Tensor x = input;
        for (SequentialImpl block : convBlocks) {
            x = block.forward(x);
        }

        // Global pooling
        x = globalPool.forward(x);

        // Flatten
        x = x.view(new long[]{x.size(0), -1});

        // Classifier
        x = classifier.forward(x);

        return x;
    }

    @Override
    protected void initializeWeights() {
        if (network == null) {
            throw new IllegalStateException("Network must be built before initializing weights");
        }

        // Initialize conv blocks
        for (SequentialImpl block : convBlocks) {
            initializeBlockWeights(block);
        }

        // Initialize classifier
        initializeBlockWeights(classifier);

        System.out.println("Generic3DCNN weights initialized");
    }

    /**
     * Initialize weights for a sequential block
     */
    private void initializeBlockWeights(SequentialImpl block) {
        NamedModuleIterator modules = block.named_modules();

        while (modules.hasNext()) {
            NamedModule namedModule = modules.next();
            Module module = namedModule.value();

            // Initialize Conv3d layers
            if (module instanceof Conv3dImpl) {
                Conv3dImpl conv = (Conv3dImpl) module;
                Tensor weight = conv.weight();

                // Kaiming initialization
                kaiming_normal_(weight, 0.0, FanModeType.FAN_IN, NonlinearityType.ReLU);

                if (conv.bias() != null && !conv.bias().isNull()) {
                    zeros_(conv.bias());
                }
            }

            // Initialize Linear layers
            if (module instanceof LinearImpl) {
                LinearImpl linear = (LinearImpl) module;
                Tensor weight = linear.weight();

                kaiming_normal_(weight, 0.0, FanModeType.FAN_IN, NonlinearityType.ReLU);

                if (linear.bias() != null && !linear.bias().isNull()) {
                    zeros_(linear.bias());
                }
            }
        }
    }

    /**
     * Get architecture description
     */
    public String getArchitectureDescription() {
        StringBuilder sb = new StringBuilder();
        sb.append("Generic3DCNN Architecture:\n");
        sb.append("==========================\n");
        sb.append(String.format("Input: [batch, %d, %d, %d, %d]\n",
                               inputChannels, inputShape[0], inputShape[1], inputShape[2]));
        sb.append(String.format("Activation: %s\n", activation));
        sb.append(String.format("Batch Normalization: %s\n", useBatchNorm ? "Yes" : "No"));
        sb.append(String.format("Dropout Rate: %.2f\n", dropoutRate));

        sb.append("\nConvolutional Blocks:\n");
        int inCh = inputChannels;
        for (int i = 0; i < blockChannels.length; i++) {
            sb.append(String.format("  Block %d: %d → %d channels (Conv3d %d×%d×%d → ",
                                   i + 1, inCh, blockChannels[i],
                                   kernelSizes[i], kernelSizes[i], kernelSizes[i]));
            if (useBatchNorm) sb.append("BN → ");
            sb.append(String.format("%s", activation));
            if (poolingSizes[i] > 1) {
                sb.append(String.format(" → MaxPool %d×%d×%d", poolingSizes[i], poolingSizes[i], poolingSizes[i]));
            }
            sb.append(")\n");
            inCh = blockChannels[i];
        }

        sb.append("\nClassifier:\n");
        sb.append("  Global Average Pooling (1×1×1)\n");
        int inFeatures = blockChannels[blockChannels.length - 1];
        for (int i = 0; i < fcLayers.length; i++) {
            sb.append(String.format("  FC: %d → %d → %s", inFeatures, fcLayers[i], activation));
            if (dropoutRate > 0) {
                sb.append(String.format(" → Dropout(%.2f)", dropoutRate));
            }
            sb.append("\n");
            inFeatures = fcLayers[i];
        }
        sb.append(String.format("  FC: %d → %d (output logits)\n", inFeatures, numClasses));

        if (isBuilt) {
            sb.append(String.format("\nTotal Parameters: %,d\n", getParameterCount()));
            sb.append(String.format("Memory Footprint: %.2f MB\n", getMemoryFootprintMB()));
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("Generic3DCNN[blocks=%d, input=%dx%dx%dx%d, classes=%d, params=%s]",
                           blockChannels.length, inputChannels, inputShape[0], inputShape[1], inputShape[2],
                           numClasses, isBuilt ? String.format("%,d", getParameterCount()) : "not built");
    }

    /**
     * Builder for Generic3DCNN
     */
    public static class Builder {
        // Required parameters
        private int inputChannels;
        private int numClasses;
        private int[] inputShape;

        // Optional parameters with defaults
        private String modelName = "Generic3DCNN";
        private int[] blockChannels = {32, 64, 128};
        private int[] kernelSizes = {3, 3, 3};
        private int[] poolingSizes = {2, 2, 2};
        private ActivationType activation = ActivationType.RELU;
        private boolean useBatchNorm = true;
        private double dropoutRate = 0.5;
        private int[] fcLayers = {256, 128};

        /**
         * Required: Set input channels
         */
        public Builder(int inputChannels, int numClasses, int[] inputShape) {
            if (inputShape == null || inputShape.length != 3) {
                throw new IllegalArgumentException("Input shape must be [depth, height, width]");
            }
            this.inputChannels = inputChannels;
            this.numClasses = numClasses;
            this.inputShape = inputShape;
        }

        /**
         * Set model name
         */
        public Builder modelName(String name) {
            this.modelName = name;
            return this;
        }

        /**
         * Set channel progression for convolutional blocks
         */
        public Builder blockChannels(int... channels) {
            if (channels.length == 0) {
                throw new IllegalArgumentException("Must specify at least one block");
            }
            this.blockChannels = channels;
            return this;
        }

        /**
         * Set kernel sizes for each block (must match number of blocks)
         */
        public Builder kernelSizes(int... sizes) {
            this.kernelSizes = sizes;
            return this;
        }

        /**
         * Set pooling sizes for each block (must match number of blocks)
         */
        public Builder poolingSizes(int... sizes) {
            this.poolingSizes = sizes;
            return this;
        }

        /**
         * Set activation function
         */
        public Builder activation(ActivationType activation) {
            this.activation = activation;
            return this;
        }

        /**
         * Enable/disable batch normalization
         */
        public Builder useBatchNorm(boolean use) {
            this.useBatchNorm = use;
            return this;
        }

        /**
         * Set dropout rate (0 = no dropout)
         */
        public Builder dropoutRate(double rate) {
            if (rate < 0 || rate >= 1) {
                throw new IllegalArgumentException("Dropout rate must be in [0, 1)");
            }
            this.dropoutRate = rate;
            return this;
        }

        /**
         * Set fully connected layer sizes (hidden layers only, output is automatic)
         */
        public Builder fcLayers(int... sizes) {
            this.fcLayers = sizes;
            return this;
        }

        /**
         * Build the model
         */
        public Generic3DCNN build() {
            // Validate configuration
            if (kernelSizes.length != blockChannels.length) {
                // Auto-fill kernel sizes if not specified
                int[] newKernelSizes = new int[blockChannels.length];
                for (int i = 0; i < newKernelSizes.length; i++) {
                    newKernelSizes[i] = (i < kernelSizes.length) ? kernelSizes[i] : 3;
                }
                kernelSizes = newKernelSizes;
            }

            if (poolingSizes.length != blockChannels.length) {
                // Auto-fill pooling sizes if not specified
                int[] newPoolingSizes = new int[blockChannels.length];
                for (int i = 0; i < newPoolingSizes.length; i++) {
                    newPoolingSizes[i] = (i < poolingSizes.length) ? poolingSizes[i] : 2;
                }
                poolingSizes = newPoolingSizes;
            }

            return new Generic3DCNN(this);
        }
    }

    /**
     * Create a lightweight model (fewer parameters)
     */
    public static Generic3DCNN createLightweight(int inputChannels, int numClasses, int[] inputShape) {
        return new Builder(inputChannels, numClasses, inputShape)
            .modelName("Lightweight3DCNN")
            .blockChannels(16, 32, 64)
            .fcLayers(128)
            .dropoutRate(0.3)
            .build();
    }

    /**
     * Create a deep model (more layers)
     */
    public static Generic3DCNN createDeep(int inputChannels, int numClasses, int[] inputShape) {
        return new Builder(inputChannels, numClasses, inputShape)
            .modelName("Deep3DCNN")
            .blockChannels(32, 64, 128, 256, 512)
            .fcLayers(512, 256, 128)
            .dropoutRate(0.5)
            .build();
    }

    /**
     * Create a wide model (more channels)
     */
    public static Generic3DCNN createWide(int inputChannels, int numClasses, int[] inputShape) {
        return new Builder(inputChannels, numClasses, inputShape)
            .modelName("Wide3DCNN")
            .blockChannels(64, 128, 256, 512)
            .fcLayers(512, 256)
            .dropoutRate(0.5)
            .build();
    }

    /**
     * Create NephNet3D-like configuration
     */
    public static Generic3DCNN createNephNetStyle(int inputChannels, int numClasses, int[] inputShape) {
        return new Builder(inputChannels, numClasses, inputShape)
            .modelName("NephNetStyle3DCNN")
            .blockChannels(32, 64, 128, 256)
            .kernelSizes(3, 3, 3, 3)
            .poolingSizes(2, 2, 2, 2)
            .activation(ActivationType.LEAKY_RELU)
            .useBatchNorm(true)
            .fcLayers(256, 128)
            .dropoutRate(0.5)
            .build();
    }
}
