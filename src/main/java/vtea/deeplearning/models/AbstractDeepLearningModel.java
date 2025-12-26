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
import vtea.deeplearning.DeepLearningConfig;
import vtea.deeplearning.data.ClassDefinition;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Abstract base class for all deep learning models in VTEA.
 * Provides common functionality for model management, persistence, and inference.
 *
 * Subclasses must implement:
 * - buildArchitecture(): Define the network structure
 * - forward(): Implement the forward pass
 * - initializeWeights(): Initialize model parameters
 *
 * @author VTEA Deep Learning Team
 */
public abstract class AbstractDeepLearningModel implements Serializable {

    private static final long serialVersionUID = 1L;

    // Model metadata
    protected String modelName;
    protected int inputChannels;
    protected int numClasses;
    protected int[] inputShape;  // [depth, height, width]

    // Class definitions for human-readable results
    protected HashMap<Integer, ClassDefinition> classDefinitions;

    // PyTorch components (transient - not serialized)
    protected transient Module network;
    protected transient Device device;
    protected transient boolean isBuilt;

    /**
     * Constructor
     */
    public AbstractDeepLearningModel(String modelName, int inputChannels, int numClasses, int[] inputShape) {
        if (inputShape == null || inputShape.length != 3) {
            throw new IllegalArgumentException("Input shape must be [depth, height, width]");
        }
        if (inputChannels < 1) {
            throw new IllegalArgumentException("Input channels must be >= 1");
        }
        if (numClasses < 2) {
            throw new IllegalArgumentException("Number of classes must be >= 2");
        }

        this.modelName = modelName;
        this.inputChannels = inputChannels;
        this.numClasses = numClasses;
        this.inputShape = inputShape;
        this.classDefinitions = new HashMap<>();
        this.isBuilt = false;

        // Initialize device
        DeepLearningConfig config = DeepLearningConfig.getInstance();
        String deviceStr = config.getDevice();
        this.device = deviceStr.equals("cuda") ? new Device(DeviceType.CUDA, 0) : new Device(DeviceType.CPU);
    }

    // Abstract methods that subclasses must implement

    /**
     * Build the network architecture.
     * Subclasses should create and configure their neural network layers here.
     */
    protected abstract void buildArchitecture();

    /**
     * Forward pass through the network.
     *
     * @param input Input tensor with shape [batch, channels, depth, height, width]
     * @return Output tensor with shape [batch, numClasses]
     */
    public abstract Tensor forward(Tensor input);

    /**
     * Initialize model weights using appropriate initialization strategy.
     * Called after buildArchitecture().
     */
    protected abstract void initializeWeights();

    // Concrete methods

    /**
     * Build the model if not already built
     */
    public void build() {
        if (!isBuilt) {
            buildArchitecture();
            initializeWeights();
            toDevice(device);
            isBuilt = true;
        }
    }

    /**
     * Move model to specified device (CPU or GPU)
     */
    public void toDevice(Device device) {
        if (network != null) {
            network.to(device);
            this.device = device;
        }
    }

    /**
     * Move model to CPU
     */
    public void toCPU() {
        toDevice(new Device(DeviceType.CPU));
    }

    /**
     * Move model to GPU
     */
    public void toGPU() {
        toDevice(new Device(DeviceType.CUDA, 0));
    }

    /**
     * Set model to training mode
     */
    public void train() {
        if (network != null) {
            network.train();
        }
    }

    /**
     * Set model to evaluation mode
     */
    public void eval() {
        if (network != null) {
            network.eval();
        }
    }

    /**
     * Predict class labels for input tensor.
     * Returns the class with highest probability.
     *
     * @param input Input tensor [batch, channels, depth, height, width]
     * @return Array of predicted class IDs
     */
    public int[] predict(Tensor input) {
        if (!isBuilt) {
            build();
        }

        eval();

        // Disable gradient computation for inference
        NoGradGuard guard = new NoGradGuard();

        try {
            // Forward pass
            Tensor output = forward(input);

            // Get class with max probability
            Tensor predictions = argmax(output, 1);

            // Convert to int array
            long batchSize = predictions.size(0);
            int[] results = new int[(int) batchSize];

            LongPointer dataPtr = predictions.data_ptr_long();
            for (int i = 0; i < batchSize; i++) {
                results[i] = (int) dataPtr.get(i);
            }

            return results;

        } finally {
            guard.close();
        }
    }

    /**
     * Predict class probabilities for input tensor.
     *
     * @param input Input tensor [batch, channels, depth, height, width]
     * @return 2D array of probabilities [batch][class]
     */
    public float[][] predictProba(Tensor input) {
        if (!isBuilt) {
            build();
        }

        eval();

        NoGradGuard guard = new NoGradGuard();

        try {
            // Forward pass
            Tensor output = forward(input);

            // Apply softmax to get probabilities
            Tensor probabilities = softmax(output, 1);

            // Convert to 2D float array
            long batchSize = probabilities.size(0);
            long numClasses = probabilities.size(1);

            float[][] results = new float[(int) batchSize][(int) numClasses];

            FloatPointer dataPtr = probabilities.data_ptr_float();
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < numClasses; j++) {
                    results[i][j] = dataPtr.get(i * numClasses + j);
                }
            }

            return results;

        } finally {
            guard.close();
        }
    }

    /**
     * Predict with class names instead of IDs
     *
     * @param input Input tensor
     * @return Array of predicted class names
     */
    public String[] predictClassNames(Tensor input) {
        int[] classIds = predict(input);
        String[] classNames = new String[classIds.length];

        for (int i = 0; i < classIds.length; i++) {
            ClassDefinition def = classDefinitions.get(classIds[i]);
            classNames[i] = def != null ? def.getClassName() : "Unknown_" + classIds[i];
        }

        return classNames;
    }

    /**
     * Get detailed prediction results including class name and probability
     *
     * @param input Input tensor
     * @return Array of prediction results
     */
    public PredictionResult[] predictDetailed(Tensor input) {
        int[] classIds = predict(input);
        float[][] probabilities = predictProba(input);

        PredictionResult[] results = new PredictionResult[classIds.length];

        for (int i = 0; i < classIds.length; i++) {
            int classId = classIds[i];
            String className = getClassName(classId);
            float probability = probabilities[i][classId];

            results[i] = new PredictionResult(classId, className, probability, probabilities[i]);
        }

        return results;
    }

    /**
     * Save model state to file.
     * Saves both the model weights and metadata.
     *
     * @param filepath Path to save the model
     */
    public void save(String filepath) throws IOException {
        if (!isBuilt) {
            throw new IllegalStateException("Cannot save model that has not been built");
        }

        // Create parent directory if needed
        File file = new File(filepath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        // Save PyTorch model weights
        String weightsPath = filepath + ".pt";
        save(network, weightsPath);

        // Save metadata (class definitions, config, etc.)
        String metadataPath = filepath + ".meta";
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(metadataPath))) {
            ModelMetadata metadata = new ModelMetadata();
            metadata.modelName = this.modelName;
            metadata.inputChannels = this.inputChannels;
            metadata.numClasses = this.numClasses;
            metadata.inputShape = this.inputShape;
            metadata.classDefinitions = this.classDefinitions;
            metadata.modelClass = this.getClass().getName();

            oos.writeObject(metadata);
        }

        System.out.println("Model saved to: " + filepath);
    }

    /**
     * Load model state from file.
     *
     * @param filepath Path to load the model from
     */
    public void load(String filepath) throws IOException {
        // Load metadata first
        String metadataPath = filepath + ".meta";
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(metadataPath))) {
            ModelMetadata metadata = (ModelMetadata) ois.readObject();

            // Verify compatibility
            if (!this.getClass().getName().equals(metadata.modelClass)) {
                throw new IOException("Model class mismatch: expected " + this.getClass().getName() +
                                    ", found " + metadata.modelClass);
            }

            // Restore metadata
            this.modelName = metadata.modelName;
            this.inputChannels = metadata.inputChannels;
            this.numClasses = metadata.numClasses;
            this.inputShape = metadata.inputShape;
            this.classDefinitions = metadata.classDefinitions;

        } catch (ClassNotFoundException e) {
            throw new IOException("Failed to load model metadata", e);
        }

        // Build architecture
        if (!isBuilt) {
            build();
        }

        // Load PyTorch weights
        String weightsPath = filepath + ".pt";
        load(network, weightsPath);

        System.out.println("Model loaded from: " + filepath);
    }

    /**
     * Get number of trainable parameters in the model
     */
    public long getParameterCount() {
        if (!isBuilt) {
            build();
        }

        long count = 0;
        NamedParameterIterator params = network.named_parameters();

        while (params.hasNext()) {
            Tensor param = params.next().value();
            long[] shape = new long[(int) param.dim()];
            for (int i = 0; i < shape.length; i++) {
                shape[i] = param.size(i);
            }

            long paramCount = 1;
            for (long dim : shape) {
                paramCount *= dim;
            }
            count += paramCount;
        }

        return count;
    }

    /**
     * Get memory footprint estimate in MB
     */
    public double getMemoryFootprintMB() {
        long paramCount = getParameterCount();
        // Assume float32 (4 bytes per parameter)
        return (paramCount * 4.0) / (1024.0 * 1024.0);
    }

    // Class definition management

    /**
     * Set class definitions for human-readable results
     */
    public void setClassDefinitions(HashMap<Integer, ClassDefinition> classDefinitions) {
        if (classDefinitions != null && classDefinitions.size() != numClasses) {
            throw new IllegalArgumentException("Number of class definitions must match numClasses");
        }
        this.classDefinitions = classDefinitions != null ? classDefinitions : new HashMap<>();
    }

    /**
     * Get class definitions
     */
    public HashMap<Integer, ClassDefinition> getClassDefinitions() {
        return classDefinitions;
    }

    /**
     * Get class name for a given ID
     */
    public String getClassName(int classId) {
        ClassDefinition def = classDefinitions.get(classId);
        return def != null ? def.getClassName() : "Class_" + classId;
    }

    /**
     * Set class name for a given ID
     */
    public void setClassName(int classId, String name) {
        ClassDefinition def = classDefinitions.get(classId);
        if (def != null) {
            def.setClassName(name);
        } else {
            classDefinitions.put(classId, new ClassDefinition(classId, name));
        }
    }

    // Getters

    public String getModelName() {
        return modelName;
    }

    public int getInputChannels() {
        return inputChannels;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public int[] getInputShape() {
        return inputShape;
    }

    public Device getDevice() {
        return device;
    }

    public boolean isBuilt() {
        return isBuilt;
    }

    public Module getNetwork() {
        return network;
    }

    @Override
    public String toString() {
        return String.format("%s[name=%s, input=%dx%dx%dx%d, classes=%d, params=%d, device=%s]",
                           getClass().getSimpleName(), modelName,
                           inputChannels, inputShape[0], inputShape[1], inputShape[2],
                           numClasses, isBuilt ? getParameterCount() : 0,
                           device != null ? device.str().getString() : "unknown");
    }

    /**
     * Container for prediction results
     */
    public static class PredictionResult {
        public final int classId;
        public final String className;
        public final float probability;
        public final float[] allProbabilities;

        public PredictionResult(int classId, String className, float probability, float[] allProbabilities) {
            this.classId = classId;
            this.className = className;
            this.probability = probability;
            this.allProbabilities = allProbabilities;
        }

        @Override
        public String toString() {
            return String.format("PredictionResult[class=%s (id=%d), prob=%.3f]",
                               className, classId, probability);
        }
    }

    /**
     * Container for model metadata (for serialization)
     */
    private static class ModelMetadata implements Serializable {
        private static final long serialVersionUID = 1L;

        String modelName;
        int inputChannels;
        int numClasses;
        int[] inputShape;
        HashMap<Integer, ClassDefinition> classDefinitions;
        String modelClass;
    }
}
