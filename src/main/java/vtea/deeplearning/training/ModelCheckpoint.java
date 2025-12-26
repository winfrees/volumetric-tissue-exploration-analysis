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

import vtea.deeplearning.models.AbstractDeepLearningModel;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * ModelCheckpoint callback for saving models during training.
 * Implements advanced checkpoint management including:
 * - Save best model based on validation metric
 * - Keep only top-K models
 * - Save training configuration with model
 * - Automatic cleanup of old checkpoints
 *
 * @author VTEA Deep Learning Team
 */
public class ModelCheckpoint implements Trainer.TrainingCallback {

    private final AbstractDeepLearningModel model;
    private final String checkpointDir;
    private final String modelName;
    private final boolean saveOnlyBest;
    private final int keepTopK;  // Keep only top K models (0 = keep all)
    private final String monitorMetric;  // "val_acc" or "val_bal_acc"
    private final boolean verbose;

    // State tracking
    private double bestMetric;
    private int bestEpoch;
    private String bestModelPath;
    private List<CheckpointInfo> savedCheckpoints;

    /**
     * Checkpoint information
     */
    private static class CheckpointInfo implements Comparable<CheckpointInfo> {
        public final int epoch;
        public final double metric;
        public final String path;

        public CheckpointInfo(int epoch, double metric, String path) {
            this.epoch = epoch;
            this.metric = metric;
            this.path = path;
        }

        @Override
        public int compareTo(CheckpointInfo other) {
            return Double.compare(other.metric, this.metric);  // Descending order
        }
    }

    /**
     * Constructor with default settings
     */
    public ModelCheckpoint(AbstractDeepLearningModel model, String checkpointDir, String modelName) {
        this(model, checkpointDir, modelName, true, 5, "val_bal_acc", true);
    }

    /**
     * Full constructor
     */
    public ModelCheckpoint(AbstractDeepLearningModel model, String checkpointDir, String modelName,
                          boolean saveOnlyBest, int keepTopK, String monitorMetric, boolean verbose) {
        this.model = model;
        this.checkpointDir = checkpointDir;
        this.modelName = modelName;
        this.saveOnlyBest = saveOnlyBest;
        this.keepTopK = keepTopK;
        this.monitorMetric = monitorMetric;
        this.verbose = verbose;

        this.bestMetric = Double.NEGATIVE_INFINITY;
        this.bestEpoch = -1;
        this.savedCheckpoints = new ArrayList<>();

        // Create checkpoint directory
        File dir = new File(checkpointDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }

    @Override
    public void onEpochStart(int epoch) {
        // Nothing to do at epoch start
    }

    @Override
    public void onEpochEnd(int epoch, double trainLoss, double trainAcc,
                          double valLoss, double valAcc, double valBalAcc) {
        try {
            // Determine metric value based on monitoring metric
            double currentMetric;
            switch (monitorMetric) {
                case "val_acc":
                    currentMetric = valAcc;
                    break;
                case "val_bal_acc":
                    currentMetric = valBalAcc;
                    break;
                case "val_loss":
                    currentMetric = -valLoss;  // Negate so higher is better
                    break;
                default:
                    currentMetric = valBalAcc;
            }

            // Check if we should save
            boolean shouldSave = false;

            if (saveOnlyBest) {
                if (currentMetric > bestMetric) {
                    bestMetric = currentMetric;
                    bestEpoch = epoch;
                    shouldSave = true;

                    if (verbose) {
                        System.out.printf("Epoch %d: %s improved from %.4f to %.4f\n",
                                        epoch + 1, monitorMetric, bestMetric, currentMetric);
                    }
                }
            } else {
                shouldSave = true;
            }

            // Save checkpoint
            if (shouldSave) {
                String filename = saveOnlyBest ?
                                String.format("%s_best", modelName) :
                                String.format("%s_epoch_%03d", modelName, epoch);

                String modelPath = checkpointDir + File.separator + filename;

                // Save model
                model.save(modelPath);

                // Save training info
                saveTrainingInfo(modelPath, epoch, trainLoss, trainAcc, valLoss, valAcc, valBalAcc);

                // Track checkpoint
                CheckpointInfo info = new CheckpointInfo(epoch, currentMetric, modelPath);
                savedCheckpoints.add(info);

                if (saveOnlyBest) {
                    bestModelPath = modelPath;
                }

                if (verbose) {
                    System.out.printf("Saved checkpoint: %s (metric: %.4f)\n", modelPath, currentMetric);
                }

                // Clean up old checkpoints if needed
                if (keepTopK > 0 && savedCheckpoints.size() > keepTopK) {
                    cleanupOldCheckpoints();
                }
            }

        } catch (Exception e) {
            System.err.println("Error saving checkpoint: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Override
    public void onBatchEnd(int epoch, int batch, int totalBatches, double loss, double acc) {
        // Nothing to do at batch end
    }

    @Override
    public void onTrainingComplete(Metrics.History history) {
        if (verbose) {
            System.out.println("\n" + "=".repeat(80));
            System.out.println("Checkpoint Summary");
            System.out.println("=".repeat(80));
            System.out.printf("Best %s: %.4f at epoch %d\n", monitorMetric, bestMetric, bestEpoch + 1);
            if (bestModelPath != null) {
                System.out.println("Best model saved at: " + bestModelPath);
            }
            System.out.println("Total checkpoints saved: " + savedCheckpoints.size());
            System.out.println("=".repeat(80));
        }

        // Save final summary
        try {
            saveFinalSummary(history);
        } catch (IOException e) {
            System.err.println("Error saving final summary: " + e.getMessage());
        }
    }

    /**
     * Save training information with checkpoint
     */
    private void saveTrainingInfo(String modelPath, int epoch,
                                  double trainLoss, double trainAcc,
                                  double valLoss, double valAcc, double valBalAcc) throws IOException {
        String infoPath = modelPath + ".info";

        Properties props = new Properties();
        props.setProperty("epoch", String.valueOf(epoch));
        props.setProperty("train_loss", String.format("%.6f", trainLoss));
        props.setProperty("train_accuracy", String.format("%.6f", trainAcc));
        props.setProperty("validation_loss", String.format("%.6f", valLoss));
        props.setProperty("validation_accuracy", String.format("%.6f", valAcc));
        props.setProperty("validation_balanced_accuracy", String.format("%.6f", valBalAcc));
        props.setProperty("timestamp", new Date().toString());

        try (FileOutputStream fos = new FileOutputStream(infoPath)) {
            props.store(fos, "Training checkpoint information");
        }
    }

    /**
     * Save final training summary
     */
    private void saveFinalSummary(Metrics.History history) throws IOException {
        String summaryPath = checkpointDir + File.separator + modelName + "_summary.txt";

        try (PrintWriter writer = new PrintWriter(new FileWriter(summaryPath))) {
            writer.println("Training Summary for " + modelName);
            writer.println("=".repeat(80));
            writer.println();

            writer.println("Best Results:");
            writer.printf("  Best %s: %.4f at epoch %d\n", monitorMetric, bestMetric, bestEpoch + 1);
            writer.printf("  Best validation accuracy: %.4f\n", history.getBestValidationAccuracy());
            writer.printf("  Best validation balanced accuracy: %.4f\n", history.getBestValidationBalancedAccuracy());
            writer.println();

            writer.println("Final Results:");
            List<Double> trainLoss = history.getTrainLoss();
            List<Double> valLoss = history.getValidationLoss();
            List<Double> valAcc = history.getValidationAccuracy();
            List<Double> valBalAcc = history.getValidationBalancedAccuracy();

            if (!trainLoss.isEmpty()) {
                int lastEpoch = trainLoss.size() - 1;
                writer.printf("  Final train loss: %.4f\n", trainLoss.get(lastEpoch));
                writer.printf("  Final validation loss: %.4f\n", valLoss.get(lastEpoch));
                writer.printf("  Final validation accuracy: %.4f\n", valAcc.get(lastEpoch));
                writer.printf("  Final validation balanced accuracy: %.4f\n", valBalAcc.get(lastEpoch));
            }
            writer.println();

            writer.println("Training History:");
            writer.println(String.format("%-8s %-12s %-12s %-12s %-12s %-12s",
                                       "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Val Bal Acc"));
            writer.println("-".repeat(80));

            for (int i = 0; i < trainLoss.size(); i++) {
                writer.println(String.format("%-8d %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f",
                                           i + 1,
                                           trainLoss.get(i),
                                           history.getTrainAccuracy().get(i),
                                           valLoss.get(i),
                                           valAcc.get(i),
                                           valBalAcc.get(i)));
            }
            writer.println();

            writer.println("Saved Checkpoints:");
            for (CheckpointInfo info : savedCheckpoints) {
                writer.printf("  Epoch %d: %s (metric: %.4f)\n", info.epoch + 1, info.path, info.metric);
            }
        }

        if (verbose) {
            System.out.println("Training summary saved to: " + summaryPath);
        }
    }

    /**
     * Clean up old checkpoints, keeping only top-K
     */
    private void cleanupOldCheckpoints() {
        // Sort checkpoints by metric
        Collections.sort(savedCheckpoints);

        // Remove checkpoints beyond top-K
        while (savedCheckpoints.size() > keepTopK) {
            CheckpointInfo toRemove = savedCheckpoints.remove(savedCheckpoints.size() - 1);

            try {
                // Delete model files
                deleteCheckpointFiles(toRemove.path);

                if (verbose) {
                    System.out.printf("Removed old checkpoint: %s (metric: %.4f)\n",
                                    toRemove.path, toRemove.metric);
                }
            } catch (IOException e) {
                System.err.println("Failed to delete checkpoint: " + toRemove.path);
            }
        }
    }

    /**
     * Delete all files associated with a checkpoint
     */
    private void deleteCheckpointFiles(String basePath) throws IOException {
        // Delete .pt file (weights)
        Files.deleteIfExists(Paths.get(basePath + ".pt"));

        // Delete .meta file (metadata)
        Files.deleteIfExists(Paths.get(basePath + ".meta"));

        // Delete .info file (training info)
        Files.deleteIfExists(Paths.get(basePath + ".info"));
    }

    /**
     * Get best model path
     */
    public String getBestModelPath() {
        return bestModelPath;
    }

    /**
     * Get best metric value
     */
    public double getBestMetric() {
        return bestMetric;
    }

    /**
     * Get best epoch
     */
    public int getBestEpoch() {
        return bestEpoch;
    }

    /**
     * Get list of saved checkpoints
     */
    public List<CheckpointInfo> getSavedCheckpoints() {
        return new ArrayList<>(savedCheckpoints);
    }

    /**
     * Load best model
     */
    public void loadBestModel() throws IOException {
        if (bestModelPath == null) {
            throw new IllegalStateException("No best model available");
        }
        model.load(bestModelPath);
        if (verbose) {
            System.out.println("Loaded best model from: " + bestModelPath);
        }
    }
}
