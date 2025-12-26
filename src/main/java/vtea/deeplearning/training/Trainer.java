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

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import vtea.deeplearning.models.AbstractDeepLearningModel;

import java.io.File;
import java.io.IOException;

/**
 * Trainer for deep learning models.
 * Handles the complete training loop including optimization, validation,
 * early stopping, and checkpointing.
 *
 * @author VTEA Deep Learning Team
 */
public class Trainer {

    private final AbstractDeepLearningModel model;
    private final TrainingConfig config;
    private final DataLoader trainLoader;
    private final DataLoader validationLoader;

    // Optimizer and scheduler
    private Optimizer optimizer;
    private LRScheduler lrScheduler;

    // Training state
    private Metrics.History history;
    private double currentLR;
    private int epochsWithoutImprovement;
    private double bestValidationMetric;
    private String bestModelPath;

    // Callbacks
    private TrainingCallback callback;

    /**
     * Interface for training callbacks
     */
    public interface TrainingCallback {
        void onEpochStart(int epoch);
        void onEpochEnd(int epoch, double trainLoss, double trainAcc,
                       double valLoss, double valAcc, double valBalAcc);
        void onBatchEnd(int epoch, int batch, int totalBatches, double loss, double acc);
        void onTrainingComplete(Metrics.History history);
    }

    /**
     * Constructor
     */
    public Trainer(AbstractDeepLearningModel model, TrainingConfig config,
                  DataLoader trainLoader, DataLoader validationLoader) {
        this.model = model;
        this.config = config;
        this.trainLoader = trainLoader;
        this.validationLoader = validationLoader;

        this.history = new Metrics.History();
        this.currentLR = config.getLearningRate();
        this.epochsWithoutImprovement = 0;
        this.bestValidationMetric = 0.0;

        initializeOptimizer();
        initializeLRScheduler();
    }

    /**
     * Initialize optimizer based on configuration
     */
    private void initializeOptimizer() {
        if (!model.isBuilt()) {
            model.build();
        }

        switch (config.getOptimizer()) {
            case SGD:
                SGDOptions sgdOpts = new SGDOptions(config.getLearningRate());
                sgdOpts.momentum().put(config.getMomentum());
                sgdOpts.weight_decay().put(config.getWeightDecay());
                optimizer = new SGD(model.getNetwork().parameters(), sgdOpts);
                break;

            case ADAM:
                AdamOptions adamOpts = new AdamOptions(config.getLearningRate());
                adamOpts.weight_decay().put(config.getWeightDecay());
                adamOpts.betas().put(new double[]{config.getBeta1(), config.getBeta2()});
                optimizer = new Adam(model.getNetwork().parameters(), adamOpts);
                break;

            case ADAMW:
                AdamWOptions adamwOpts = new AdamWOptions(config.getLearningRate());
                adamwOpts.weight_decay().put(config.getWeightDecay());
                adamwOpts.betas().put(new double[]{config.getBeta1(), config.getBeta2()});
                optimizer = new AdamW(model.getNetwork().parameters(), adamwOpts);
                break;

            case RMSPROP:
                RMSpropOptions rmspropOpts = new RMSpropOptions(config.getLearningRate());
                rmspropOpts.weight_decay().put(config.getWeightDecay());
                rmspropOpts.momentum().put(config.getMomentum());
                optimizer = new RMSprop(model.getNetwork().parameters(), rmspropOpts);
                break;

            default:
                throw new IllegalArgumentException("Unsupported optimizer: " + config.getOptimizer());
        }

        if (config.isVerbose()) {
            System.out.println("Initialized optimizer: " + config.getOptimizer());
        }
    }

    /**
     * Initialize learning rate scheduler
     */
    private void initializeLRScheduler() {
        if (!config.isUseLRScheduler()) {
            lrScheduler = null;
            return;
        }

        switch (config.getLrSchedulerType()) {
            case STEP:
                StepLROptions stepOpts = new StepLROptions(config.getLrDecayEpochs());
                stepOpts.gamma(config.getLrDecayFactor());
                lrScheduler = new StepLR(optimizer, stepOpts);
                break;

            case EXPONENTIAL:
                ExponentialLROptions expOpts = new ExponentialLROptions(config.getLrDecayFactor());
                lrScheduler = new ExponentialLR(optimizer, expOpts);
                break;

            default:
                lrScheduler = null;
                if (config.isVerbose()) {
                    System.out.println("Warning: LR scheduler type " + config.getLrSchedulerType() +
                                     " not fully implemented, using step decay");
                    StepLROptions defaultOpts = new StepLROptions(config.getLrDecayEpochs());
                    defaultOpts.gamma(config.getLrDecayFactor());
                    lrScheduler = new StepLR(optimizer, defaultOpts);
                }
        }

        if (config.isVerbose() && lrScheduler != null) {
            System.out.println("Initialized LR scheduler: " + config.getLrSchedulerType());
        }
    }

    /**
     * Set training callback
     */
    public void setCallback(TrainingCallback callback) {
        this.callback = callback;
    }

    /**
     * Train the model
     */
    public Metrics.History train() throws IOException {
        if (config.isVerbose()) {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Starting Training");
            System.out.println("=".repeat(80));
            System.out.println("Model: " + model.getModelName());
            System.out.println("Training samples: " + trainLoader.getNumSamples());
            System.out.println("Validation samples: " + validationLoader.getNumSamples());
            System.out.println("Epochs: " + config.getEpochs());
            System.out.println("Batch size: " + config.getBatchSize());
            System.out.println("Initial learning rate: " + config.getLearningRate());
            System.out.println("Optimizer: " + config.getOptimizer());
            System.out.println("=".repeat(80) + "\n");
        }

        // Create checkpoint directory
        if (config.isSaveCheckpoints()) {
            File checkpointDir = new File(config.getCheckpointDir());
            if (!checkpointDir.exists()) {
                checkpointDir.mkdirs();
            }
        }

        // Training loop
        for (int epoch = 0; epoch < config.getEpochs(); epoch++) {
            if (callback != null) {
                callback.onEpochStart(epoch);
            }

            // Training phase
            double trainLoss = trainEpoch(epoch);
            double trainAcc = evaluateAccuracy(trainLoader);

            // Validation phase
            double valLoss = 0;
            double valAcc = 0;
            double valBalAcc = 0;

            if (epoch % config.getValidationFrequency() == 0) {
                valLoss = validateEpoch();
                valAcc = evaluateAccuracy(validationLoader);
                valBalAcc = evaluateBalancedAccuracy(validationLoader);
            }

            // Update learning rate
            if (lrScheduler != null) {
                lrScheduler.step();
                currentLR = getCurrentLearningRate();
            }

            // Record history
            history.recordEpoch(trainLoss, trainAcc, valLoss, valAcc, valBalAcc, currentLR);

            // Print epoch summary
            if (config.isVerbose()) {
                System.out.printf("Epoch %d/%d - Loss: %.4f - Acc: %.4f - Val Loss: %.4f - Val Acc: %.4f - Val Bal Acc: %.4f - LR: %.6f\n",
                                epoch + 1, config.getEpochs(), trainLoss, trainAcc, valLoss, valAcc, valBalAcc, currentLR);
            }

            if (callback != null) {
                callback.onEpochEnd(epoch, trainLoss, trainAcc, valLoss, valAcc, valBalAcc);
            }

            // Checkpointing
            if (config.isSaveCheckpoints()) {
                saveCheckpoint(epoch, valBalAcc);
            }

            // Early stopping
            if (config.isUseEarlyStopping()) {
                if (checkEarlyStopping(valBalAcc)) {
                    if (config.isVerbose()) {
                        System.out.println("\nEarly stopping triggered at epoch " + (epoch + 1));
                    }
                    break;
                }
            }
        }

        // Training complete
        if (config.isVerbose()) {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Training Complete!");
            System.out.println("Best validation balanced accuracy: " + String.format("%.4f", history.getBestValidationBalancedAccuracy()));
            System.out.println("Best epoch: " + (history.getBestEpoch() + 1));
            if (bestModelPath != null) {
                System.out.println("Best model saved at: " + bestModelPath);
            }
            System.out.println("=".repeat(80) + "\n");
        }

        if (callback != null) {
            callback.onTrainingComplete(history);
        }

        return history;
    }

    /**
     * Train one epoch
     */
    private double trainEpoch(int epoch) {
        model.train();
        trainLoader.reset();

        double totalLoss = 0;
        int numBatches = 0;

        while (trainLoader.hasNext()) {
            DataLoader.Batch batch = trainLoader.nextBatch();

            try {
                // Move to device
                Tensor data = batch.data.to(model.getDevice());
                Tensor labels = batch.labels.to(model.getDevice());

                // Zero gradients
                optimizer.zero_grad();

                // Forward pass
                Tensor predictions = model.forward(data);

                // Compute loss
                Tensor loss;
                if (config.isUseClassWeights() && config.getClassWeights() != null) {
                    double lossValue = Metrics.weightedCrossEntropyLoss(predictions, labels, config.getClassWeights());
                    loss = scalar_tensor(lossValue);
                } else {
                    loss = cross_entropy_loss(predictions, labels);
                }

                // Backward pass
                loss.backward();

                // Update weights
                optimizer.step();

                // Record loss
                double lossValue = loss.item_double();
                totalLoss += lossValue;
                numBatches++;

                // Batch callback
                if (callback != null && numBatches % config.getLogFrequency() == 0) {
                    double batchAcc = Metrics.accuracy(predictions, labels);
                    callback.onBatchEnd(epoch, numBatches, trainLoader.getNumBatches(), lossValue, batchAcc);
                }

                // Cleanup
                loss.close();
                predictions.close();
                data.close();
                labels.close();

            } finally {
                batch.close();
            }
        }

        return totalLoss / numBatches;
    }

    /**
     * Validate one epoch
     */
    private double validateEpoch() {
        model.eval();
        validationLoader.reset();

        double totalLoss = 0;
        int numBatches = 0;

        NoGradGuard guard = new NoGradGuard();

        try {
            while (validationLoader.hasNext()) {
                DataLoader.Batch batch = validationLoader.nextBatch();

                try {
                    Tensor data = batch.data.to(model.getDevice());
                    Tensor labels = batch.labels.to(model.getDevice());

                    Tensor predictions = model.forward(data);
                    double lossValue = Metrics.crossEntropyLoss(predictions, labels);

                    totalLoss += lossValue;
                    numBatches++;

                    predictions.close();
                    data.close();
                    labels.close();

                } finally {
                    batch.close();
                }
            }
        } finally {
            guard.close();
        }

        return totalLoss / numBatches;
    }

    /**
     * Evaluate accuracy on a data loader
     */
    private double evaluateAccuracy(DataLoader loader) {
        model.eval();
        loader.reset();

        double totalAcc = 0;
        int numBatches = 0;

        NoGradGuard guard = new NoGradGuard();

        try {
            while (loader.hasNext()) {
                DataLoader.Batch batch = loader.nextBatch();

                try {
                    Tensor data = batch.data.to(model.getDevice());
                    Tensor labels = batch.labels.to(model.getDevice());

                    Tensor predictions = model.forward(data);
                    double acc = Metrics.accuracy(predictions, labels);

                    totalAcc += acc;
                    numBatches++;

                    predictions.close();
                    data.close();
                    labels.close();

                } finally {
                    batch.close();
                }
            }
        } finally {
            guard.close();
        }

        return totalAcc / numBatches;
    }

    /**
     * Evaluate balanced accuracy on a data loader
     */
    private double evaluateBalancedAccuracy(DataLoader loader) {
        model.eval();
        loader.reset();

        double totalBalAcc = 0;
        int numBatches = 0;

        NoGradGuard guard = new NoGradGuard();

        try {
            while (loader.hasNext()) {
                DataLoader.Batch batch = loader.nextBatch();

                try {
                    Tensor data = batch.data.to(model.getDevice());
                    Tensor labels = batch.labels.to(model.getDevice());

                    Tensor predictions = model.forward(data);
                    double balAcc = Metrics.balancedAccuracy(predictions, labels, model.getNumClasses());

                    totalBalAcc += balAcc;
                    numBatches++;

                    predictions.close();
                    data.close();
                    labels.close();

                } finally {
                    batch.close();
                }
            }
        } finally {
            guard.close();
        }

        return totalBalAcc / numBatches;
    }

    /**
     * Save checkpoint
     */
    private void saveCheckpoint(int epoch, double validationMetric) throws IOException {
        boolean shouldSave = false;

        if (config.isSaveOnlyBest()) {
            if (validationMetric > bestValidationMetric) {
                bestValidationMetric = validationMetric;
                shouldSave = true;
            }
        } else {
            if (epoch % config.getCheckpointFrequency() == 0) {
                shouldSave = true;
            }
        }

        if (shouldSave) {
            String filename = config.isSaveOnlyBest() ?
                            "best_model" :
                            String.format("checkpoint_epoch_%03d", epoch);

            bestModelPath = config.getCheckpointDir() + File.separator + filename;
            model.save(bestModelPath);

            if (config.isVerbose()) {
                System.out.println("Saved checkpoint: " + bestModelPath);
            }
        }
    }

    /**
     * Check early stopping condition
     */
    private boolean checkEarlyStopping(double validationMetric) {
        if (validationMetric > bestValidationMetric + config.getEarlyStoppingMinDelta()) {
            bestValidationMetric = validationMetric;
            epochsWithoutImprovement = 0;
            return false;
        } else {
            epochsWithoutImprovement++;
            return epochsWithoutImprovement >= config.getEarlyStoppingPatience();
        }
    }

    /**
     * Get current learning rate
     */
    private double getCurrentLearningRate() {
        // Access learning rate from optimizer
        TensorArrayRef paramGroups = optimizer.param_groups();
        if (paramGroups.size() > 0) {
            OptimizerParamGroup group = paramGroups.get(0);
            TensorDict options = group.options();
            Scalar lrScalar = options.find("lr");
            if (lrScalar != null && !lrScalar.isNull()) {
                return lrScalar.toDouble();
            }
        }
        return currentLR;
    }

    /**
     * Get training history
     */
    public Metrics.History getHistory() {
        return history;
    }

    /**
     * Get best model path
     */
    public String getBestModelPath() {
        return bestModelPath;
    }
}
