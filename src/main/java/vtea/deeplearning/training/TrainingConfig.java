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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;

/**
 * Configuration class for training parameters.
 * Supports serialization for reproducibility.
 *
 * @author VTEA Deep Learning Team
 */
public class TrainingConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    // Training parameters
    private int epochs;
    private int batchSize;
    private double learningRate;
    private double weightDecay;
    private double momentum;

    // Optimizer settings
    private OptimizerType optimizer;
    private double beta1;  // For Adam
    private double beta2;  // For Adam

    // Learning rate scheduling
    private boolean useLRScheduler;
    private LRSchedulerType lrSchedulerType;
    private double lrDecayFactor;
    private int lrDecayEpochs;  // Step decay interval
    private int lrWarmupEpochs;

    // Regularization
    private double dropoutRate;
    private boolean useAugmentation;

    // Validation
    private double validationSplit;  // Fraction of data for validation
    private int validationFrequency; // Validate every N epochs

    // Early stopping
    private boolean useEarlyStopping;
    private int earlyStoppingPatience;
    private double earlyStoppingMinDelta;

    // Checkpointing
    private boolean saveCheckpoints;
    private String checkpointDir;
    private int checkpointFrequency;
    private boolean saveOnlyBest;

    // Class balancing
    private boolean useClassWeights;
    private double[] classWeights;

    // Mixed precision training
    private boolean useMixedPrecision;

    // Logging
    private int logFrequency;  // Log every N batches
    private boolean verbose;

    /**
     * Optimizer types
     */
    public enum OptimizerType {
        SGD,
        ADAM,
        ADAMW,
        RMSPROP
    }

    /**
     * Learning rate scheduler types
     */
    public enum LRSchedulerType {
        STEP,      // Step decay
        EXPONENTIAL,
        COSINE,
        PLATEAU    // Reduce on plateau
    }

    /**
     * Default constructor with reasonable defaults
     */
    public TrainingConfig() {
        // Training defaults
        this.epochs = 100;
        this.batchSize = 16;
        this.learningRate = 0.001;
        this.weightDecay = 1e-5;
        this.momentum = 0.9;

        // Optimizer defaults
        this.optimizer = OptimizerType.ADAM;
        this.beta1 = 0.9;
        this.beta2 = 0.999;

        // LR scheduling defaults
        this.useLRScheduler = true;
        this.lrSchedulerType = LRSchedulerType.STEP;
        this.lrDecayFactor = 0.1;
        this.lrDecayEpochs = 30;
        this.lrWarmupEpochs = 0;

        // Regularization defaults
        this.dropoutRate = 0.5;
        this.useAugmentation = true;

        // Validation defaults
        this.validationSplit = 0.2;
        this.validationFrequency = 1;

        // Early stopping defaults
        this.useEarlyStopping = true;
        this.earlyStoppingPatience = 10;
        this.earlyStoppingMinDelta = 0.001;

        // Checkpointing defaults
        this.saveCheckpoints = true;
        this.checkpointDir = System.getProperty("user.home") + "/.vtea/models/checkpoints";
        this.checkpointFrequency = 5;
        this.saveOnlyBest = true;

        // Class balancing defaults
        this.useClassWeights = false;
        this.classWeights = null;

        // Mixed precision defaults
        this.useMixedPrecision = false;

        // Logging defaults
        this.logFrequency = 10;
        this.verbose = true;
    }

    /**
     * Save configuration to JSON file
     */
    public void saveToFile(String filepath) throws IOException {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(this);

        File file = new File(filepath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        try (FileWriter writer = new FileWriter(filepath)) {
            writer.write(json);
        }
    }

    /**
     * Load configuration from JSON file
     */
    public static TrainingConfig loadFromFile(String filepath) throws IOException {
        Gson gson = new Gson();
        try (FileReader reader = new FileReader(filepath)) {
            return gson.fromJson(reader, TrainingConfig.class);
        }
    }

    /**
     * Create a copy of this configuration
     */
    public TrainingConfig copy() {
        TrainingConfig copy = new TrainingConfig();

        copy.epochs = this.epochs;
        copy.batchSize = this.batchSize;
        copy.learningRate = this.learningRate;
        copy.weightDecay = this.weightDecay;
        copy.momentum = this.momentum;

        copy.optimizer = this.optimizer;
        copy.beta1 = this.beta1;
        copy.beta2 = this.beta2;

        copy.useLRScheduler = this.useLRScheduler;
        copy.lrSchedulerType = this.lrSchedulerType;
        copy.lrDecayFactor = this.lrDecayFactor;
        copy.lrDecayEpochs = this.lrDecayEpochs;
        copy.lrWarmupEpochs = this.lrWarmupEpochs;

        copy.dropoutRate = this.dropoutRate;
        copy.useAugmentation = this.useAugmentation;

        copy.validationSplit = this.validationSplit;
        copy.validationFrequency = this.validationFrequency;

        copy.useEarlyStopping = this.useEarlyStopping;
        copy.earlyStoppingPatience = this.earlyStoppingPatience;
        copy.earlyStoppingMinDelta = this.earlyStoppingMinDelta;

        copy.saveCheckpoints = this.saveCheckpoints;
        copy.checkpointDir = this.checkpointDir;
        copy.checkpointFrequency = this.checkpointFrequency;
        copy.saveOnlyBest = this.saveOnlyBest;

        copy.useClassWeights = this.useClassWeights;
        if (this.classWeights != null) {
            copy.classWeights = this.classWeights.clone();
        }

        copy.useMixedPrecision = this.useMixedPrecision;

        copy.logFrequency = this.logFrequency;
        copy.verbose = this.verbose;

        return copy;
    }

    // Getters and setters

    public int getEpochs() { return epochs; }
    public void setEpochs(int epochs) { this.epochs = epochs; }

    public int getBatchSize() { return batchSize; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }

    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }

    public double getWeightDecay() { return weightDecay; }
    public void setWeightDecay(double weightDecay) { this.weightDecay = weightDecay; }

    public double getMomentum() { return momentum; }
    public void setMomentum(double momentum) { this.momentum = momentum; }

    public OptimizerType getOptimizer() { return optimizer; }
    public void setOptimizer(OptimizerType optimizer) { this.optimizer = optimizer; }

    public double getBeta1() { return beta1; }
    public void setBeta1(double beta1) { this.beta1 = beta1; }

    public double getBeta2() { return beta2; }
    public void setBeta2(double beta2) { this.beta2 = beta2; }

    public boolean isUseLRScheduler() { return useLRScheduler; }
    public void setUseLRScheduler(boolean useLRScheduler) { this.useLRScheduler = useLRScheduler; }

    public LRSchedulerType getLrSchedulerType() { return lrSchedulerType; }
    public void setLrSchedulerType(LRSchedulerType lrSchedulerType) { this.lrSchedulerType = lrSchedulerType; }

    public double getLrDecayFactor() { return lrDecayFactor; }
    public void setLrDecayFactor(double lrDecayFactor) { this.lrDecayFactor = lrDecayFactor; }

    public int getLrDecayEpochs() { return lrDecayEpochs; }
    public void setLrDecayEpochs(int lrDecayEpochs) { this.lrDecayEpochs = lrDecayEpochs; }

    public int getLrWarmupEpochs() { return lrWarmupEpochs; }
    public void setLrWarmupEpochs(int lrWarmupEpochs) { this.lrWarmupEpochs = lrWarmupEpochs; }

    public double getDropoutRate() { return dropoutRate; }
    public void setDropoutRate(double dropoutRate) { this.dropoutRate = dropoutRate; }

    public boolean isUseAugmentation() { return useAugmentation; }
    public void setUseAugmentation(boolean useAugmentation) { this.useAugmentation = useAugmentation; }

    public double getValidationSplit() { return validationSplit; }
    public void setValidationSplit(double validationSplit) { this.validationSplit = validationSplit; }

    public int getValidationFrequency() { return validationFrequency; }
    public void setValidationFrequency(int validationFrequency) { this.validationFrequency = validationFrequency; }

    public boolean isUseEarlyStopping() { return useEarlyStopping; }
    public void setUseEarlyStopping(boolean useEarlyStopping) { this.useEarlyStopping = useEarlyStopping; }

    public int getEarlyStoppingPatience() { return earlyStoppingPatience; }
    public void setEarlyStoppingPatience(int earlyStoppingPatience) { this.earlyStoppingPatience = earlyStoppingPatience; }

    public double getEarlyStoppingMinDelta() { return earlyStoppingMinDelta; }
    public void setEarlyStoppingMinDelta(double earlyStoppingMinDelta) { this.earlyStoppingMinDelta = earlyStoppingMinDelta; }

    public boolean isSaveCheckpoints() { return saveCheckpoints; }
    public void setSaveCheckpoints(boolean saveCheckpoints) { this.saveCheckpoints = saveCheckpoints; }

    public String getCheckpointDir() { return checkpointDir; }
    public void setCheckpointDir(String checkpointDir) { this.checkpointDir = checkpointDir; }

    public int getCheckpointFrequency() { return checkpointFrequency; }
    public void setCheckpointFrequency(int checkpointFrequency) { this.checkpointFrequency = checkpointFrequency; }

    public boolean isSaveOnlyBest() { return saveOnlyBest; }
    public void setSaveOnlyBest(boolean saveOnlyBest) { this.saveOnlyBest = saveOnlyBest; }

    public boolean isUseClassWeights() { return useClassWeights; }
    public void setUseClassWeights(boolean useClassWeights) { this.useClassWeights = useClassWeights; }

    public double[] getClassWeights() { return classWeights; }
    public void setClassWeights(double[] classWeights) { this.classWeights = classWeights; }

    public boolean isUseMixedPrecision() { return useMixedPrecision; }
    public void setUseMixedPrecision(boolean useMixedPrecision) { this.useMixedPrecision = useMixedPrecision; }

    public int getLogFrequency() { return logFrequency; }
    public void setLogFrequency(int logFrequency) { this.logFrequency = logFrequency; }

    public boolean isVerbose() { return verbose; }
    public void setVerbose(boolean verbose) { this.verbose = verbose; }

    @Override
    public String toString() {
        return String.format("TrainingConfig[epochs=%d, batch=%d, lr=%.4f, optimizer=%s, scheduler=%s]",
                           epochs, batchSize, learningRate, optimizer,
                           useLRScheduler ? lrSchedulerType : "None");
    }

    /**
     * Builder for TrainingConfig
     */
    public static class Builder {
        private TrainingConfig config;

        public Builder() {
            config = new TrainingConfig();
        }

        public Builder epochs(int epochs) {
            config.epochs = epochs;
            return this;
        }

        public Builder batchSize(int batchSize) {
            config.batchSize = batchSize;
            return this;
        }

        public Builder learningRate(double lr) {
            config.learningRate = lr;
            return this;
        }

        public Builder optimizer(OptimizerType optimizer) {
            config.optimizer = optimizer;
            return this;
        }

        public Builder weightDecay(double decay) {
            config.weightDecay = decay;
            return this;
        }

        public Builder validationSplit(double split) {
            config.validationSplit = split;
            return this;
        }

        public Builder useEarlyStopping(boolean use) {
            config.useEarlyStopping = use;
            return this;
        }

        public Builder earlyStoppingPatience(int patience) {
            config.earlyStoppingPatience = patience;
            return this;
        }

        public Builder checkpointDir(String dir) {
            config.checkpointDir = dir;
            return this;
        }

        public Builder classWeights(double[] weights) {
            config.classWeights = weights;
            config.useClassWeights = true;
            return this;
        }

        public Builder verbose(boolean verbose) {
            config.verbose = verbose;
            return this;
        }

        public TrainingConfig build() {
            return config;
        }
    }
}
