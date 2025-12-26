package vtea.deeplearning.models;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;

/**
 * Configuration class for Variational Autoencoder (VAE) models.
 *
 * <p>This class encapsulates all hyperparameters and architecture choices
 * for VAE models, including:</p>
 * <ul>
 *   <li>Architecture parameters (latent dimensions, network depth)</li>
 *   <li>Training hyperparameters (learning rate, batch size, epochs)</li>
 *   <li>Loss function configuration (β-VAE weighting, KL warmup)</li>
 *   <li>Data processing settings (normalization, augmentation)</li>
 * </ul>
 *
 * <p>Supports JSON serialization for saving/loading configurations.</p>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class VAEConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    // ===== Architecture Parameters =====

    /** Size of input volumes (e.g., 64 for 64³) */
    private int inputSize = 64;

    /** Number of input channels (1 for grayscale, 3 for RGB) */
    private int numChannels = 1;

    /** Dimensionality of latent space */
    private int latentDim = 32;

    /** Encoder channel progression (e.g., [32, 64, 128, 256]) */
    private int[] encoderChannels = {32, 64, 128, 256};

    /** Predefined architecture size */
    private VAEArchitecture architecture = VAEArchitecture.MEDIUM;

    // ===== Training Hyperparameters =====

    /** Learning rate for optimizer */
    private double learningRate = 1e-4;

    /** Batch size for training */
    private int batchSize = 16;

    /** Number of training epochs */
    private int epochs = 100;

    /** β parameter for β-VAE (KL weighting) */
    private double beta = 1.0;

    /** Whether to use GPU acceleration */
    private boolean useGPU = false;

    // ===== Loss Configuration =====

    /** Type of reconstruction loss */
    private ReconstructionType reconstructionType = ReconstructionType.MSE;

    /** Whether to use KL divergence warmup */
    private boolean useKLWarmup = true;

    /** Number of warmup epochs for KL divergence */
    private int warmupEpochs = 10;

    // ===== Data Processing =====

    /** Normalization strategy */
    private NormalizationType normalization = NormalizationType.ZSCORE;

    /** Whether to use data augmentation */
    private boolean useAugmentation = true;

    /** Random seed for reproducibility */
    private long randomSeed = 42L;

    // ===== Model Variant =====

    /** Type of VAE model */
    private VAEType vaeType = VAEType.STANDARD;

    /** For conditional VAE: number of classes */
    private int numClasses = 0;

    /**
     * Predefined VAE architecture sizes.
     */
    public enum VAEArchitecture {
        /** Small: 32³ input, 16D latent, [16, 32, 64, 128] channels */
        SMALL,

        /** Medium: 64³ input, 32D latent, [32, 64, 128, 256] channels */
        MEDIUM,

        /** Large: 128³ input, 64D latent, [32, 64, 128, 256, 512] channels */
        LARGE,

        /** Custom: user-defined parameters */
        CUSTOM
    }

    /**
     * VAE model variants.
     */
    public enum VAEType {
        /** Standard VAE */
        STANDARD,

        /** β-VAE for disentangled representations */
        BETA_VAE,

        /** Conditional VAE with class labels */
        CONDITIONAL
    }

    /**
     * Reconstruction loss types.
     */
    public enum ReconstructionType {
        /** Mean Squared Error */
        MSE,

        /** Binary Cross-Entropy */
        BCE,

        /** L1 (Mean Absolute Error) */
        L1
    }

    /**
     * Normalization strategies.
     */
    public enum NormalizationType {
        /** Z-score: (x - mean) / std */
        ZSCORE,

        /** Min-max to [0, 1] */
        MINMAX,

        /** No normalization */
        NONE
    }

    /**
     * Creates a VAEConfig with default settings.
     */
    public VAEConfig() {
        // Use default values
    }

    /**
     * Creates a VAEConfig for a predefined architecture.
     *
     * @param architecture The predefined architecture to use
     */
    public VAEConfig(VAEArchitecture architecture) {
        setArchitecture(architecture);
    }

    /**
     * Sets parameters for a predefined architecture.
     *
     * @param architecture The architecture to configure
     */
    public void setArchitecture(VAEArchitecture architecture) {
        this.architecture = architecture;

        switch (architecture) {
            case SMALL:
                this.inputSize = 32;
                this.latentDim = 16;
                this.encoderChannels = new int[]{16, 32, 64, 128};
                break;

            case MEDIUM:
                this.inputSize = 64;
                this.latentDim = 32;
                this.encoderChannels = new int[]{32, 64, 128, 256};
                break;

            case LARGE:
                this.inputSize = 128;
                this.latentDim = 64;
                this.encoderChannels = new int[]{32, 64, 128, 256, 512};
                break;

            case CUSTOM:
                // Keep user-defined parameters
                break;
        }
    }

    // ===== Getters and Setters =====

    public int getInputSize() {
        return inputSize;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
        this.architecture = VAEArchitecture.CUSTOM;
    }

    public int getNumChannels() {
        return numChannels;
    }

    public void setNumChannels(int numChannels) {
        this.numChannels = numChannels;
    }

    public int getLatentDim() {
        return latentDim;
    }

    public void setLatentDim(int latentDim) {
        this.latentDim = latentDim;
        this.architecture = VAEArchitecture.CUSTOM;
    }

    public int[] getEncoderChannels() {
        return encoderChannels;
    }

    public void setEncoderChannels(int[] encoderChannels) {
        this.encoderChannels = encoderChannels;
        this.architecture = VAEArchitecture.CUSTOM;
    }

    public VAEArchitecture getArchitectureType() {
        return architecture;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public double getBeta() {
        return beta;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public boolean isUseGPU() {
        return useGPU;
    }

    public void setUseGPU(boolean useGPU) {
        this.useGPU = useGPU;
    }

    public ReconstructionType getReconstructionType() {
        return reconstructionType;
    }

    public void setReconstructionType(ReconstructionType reconstructionType) {
        this.reconstructionType = reconstructionType;
    }

    public boolean isUseKLWarmup() {
        return useKLWarmup;
    }

    public void setUseKLWarmup(boolean useKLWarmup) {
        this.useKLWarmup = useKLWarmup;
    }

    public int getWarmupEpochs() {
        return warmupEpochs;
    }

    public void setWarmupEpochs(int warmupEpochs) {
        this.warmupEpochs = warmupEpochs;
    }

    public NormalizationType getNormalization() {
        return normalization;
    }

    public void setNormalization(NormalizationType normalization) {
        this.normalization = normalization;
    }

    public boolean isUseAugmentation() {
        return useAugmentation;
    }

    public void setUseAugmentation(boolean useAugmentation) {
        this.useAugmentation = useAugmentation;
    }

    public long getRandomSeed() {
        return randomSeed;
    }

    public void setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
    }

    public VAEType getVaeType() {
        return vaeType;
    }

    public void setVaeType(VAEType vaeType) {
        this.vaeType = vaeType;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    // ===== Serialization =====

    /**
     * Converts configuration to JSON string.
     *
     * @return JSON representation
     */
    public String toJson() {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(this);
    }

    /**
     * Creates configuration from JSON string.
     *
     * @param json JSON string
     * @return VAEConfig object
     */
    public static VAEConfig fromJson(String json) {
        Gson gson = new Gson();
        return gson.fromJson(json, VAEConfig.class);
    }

    /**
     * Saves configuration to JSON file.
     *
     * @param filepath Path to save file
     * @throws IOException if save fails
     */
    public void saveToFile(String filepath) throws IOException {
        try (FileWriter writer = new FileWriter(filepath)) {
            writer.write(toJson());
        }
    }

    /**
     * Loads configuration from JSON file.
     *
     * @param filepath Path to config file
     * @return VAEConfig object
     * @throws IOException if load fails
     */
    public static VAEConfig loadFromFile(String filepath) throws IOException {
        try (FileReader reader = new FileReader(filepath)) {
            Gson gson = new Gson();
            return gson.fromJson(reader, VAEConfig.class);
        }
    }

    /**
     * Validates configuration parameters.
     *
     * @throws IllegalStateException if configuration is invalid
     */
    public void validate() {
        if (inputSize <= 0 || !isPowerOfTwo(inputSize)) {
            throw new IllegalStateException("Input size must be a positive power of 2");
        }

        if (latentDim <= 0) {
            throw new IllegalStateException("Latent dimension must be positive");
        }

        if (numChannels <= 0) {
            throw new IllegalStateException("Number of channels must be positive");
        }

        if (learningRate <= 0) {
            throw new IllegalStateException("Learning rate must be positive");
        }

        if (batchSize <= 0) {
            throw new IllegalStateException("Batch size must be positive");
        }

        if (epochs <= 0) {
            throw new IllegalStateException("Epochs must be positive");
        }

        if (beta < 0) {
            throw new IllegalStateException("Beta must be non-negative");
        }

        if (encoderChannels == null || encoderChannels.length == 0) {
            throw new IllegalStateException("Encoder channels must be specified");
        }

        if (vaeType == VAEType.CONDITIONAL && numClasses <= 0) {
            throw new IllegalStateException("Conditional VAE requires numClasses > 0");
        }
    }

    private boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    @Override
    public String toString() {
        return String.format("VAEConfig[architecture=%s, input=%d³, latent=%dD, " +
                           "channels=%d, lr=%.2e, batch=%d, beta=%.1f]",
                           architecture, inputSize, latentDim, numChannels,
                           learningRate, batchSize, beta);
    }

    /**
     * Creates a copy of this configuration.
     *
     * @return Deep copy of configuration
     */
    public VAEConfig copy() {
        return VAEConfig.fromJson(this.toJson());
    }
}
