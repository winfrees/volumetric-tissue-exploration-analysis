package vtea.deeplearning.training;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.bytedeco.pytorch.Module;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import vtea.deeplearning.models.VAEConfig;
import vtea.deeplearning.models.VariationalAutoencoder3D;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

/**
 * Handles saving and loading of trained models with metadata.
 *
 * <p>Integrates with VTEA's file persistence patterns and provides
 * checkpointing capabilities for training resumption.</p>
 *
 * <p>Saved checkpoint includes:</p>
 * <ul>
 *   <li>Model state (PyTorch parameters)</li>
 *   <li>Model configuration (VAEConfig as JSON)</li>
 *   <li>Training metrics</li>
 *   <li>Epoch number</li>
 *   <li>Timestamp and metadata</li>
 * </ul>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class VAEModelCheckpoint {

    private static final Logger logger = LoggerFactory.getLogger(ModelCheckpoint.class);

    private static final String MODEL_WEIGHTS_FILE = "model.pt";
    private static final String CONFIG_FILE = "config.json";
    private static final String METADATA_FILE = "metadata.json";
    private static final String METRICS_FILE = "metrics.csv";

    private final String checkpointDir;
    private final boolean saveOnlyBest;
    private final int keepLast; // Number of recent checkpoints to keep

    private double bestLoss;
    private int checkpointCount;

    /**
     * Creates a ModelCheckpoint handler.
     *
     * @param checkpointDir Directory to save checkpoints
     * @param saveOnlyBest Whether to save only best models
     * @param keepLast Number of recent checkpoints to keep (0 = keep all)
     */
    public VAEModelCheckpoint(String checkpointDir, boolean saveOnlyBest, int keepLast) {
        if (checkpointDir == null || checkpointDir.isEmpty()) {
            throw new IllegalArgumentException("Checkpoint directory cannot be null/empty");
        }

        this.checkpointDir = checkpointDir;
        this.saveOnlyBest = saveOnlyBest;
        this.keepLast = keepLast;
        this.bestLoss = Double.MAX_VALUE;
        this.checkpointCount = 0;

        // Create directory if doesn't exist
        try {
            Files.createDirectories(Paths.get(checkpointDir));
            logger.info("ModelCheckpoint initialized: dir={}, save_only_best={}, keep_last={}",
                       checkpointDir, saveOnlyBest, keepLast);
        } catch (IOException e) {
            logger.error("Failed to create checkpoint directory: {}", checkpointDir, e);
            throw new RuntimeException("Checkpoint directory creation failed", e);
        }
    }

    /**
     * Creates a ModelCheckpoint with default settings.
     *
     * @param checkpointDir Directory to save checkpoints
     */
    public VAEModelCheckpoint(String checkpointDir) {
        this(checkpointDir, false, 3); // Keep last 3 checkpoints by default
    }

    /**
     * Saves a model checkpoint.
     *
     * @param model The VAE model to save
     * @param config Model configuration
     * @param epoch Current epoch
     * @param valLoss Validation loss
     * @param metrics Training metrics
     * @return Path to saved checkpoint
     */
    public String save(VariationalAutoencoder3D model,
                      VAEConfig config,
                      int epoch,
                      double valLoss,
                      TrainingMetrics metrics) {

        // Check if should save
        boolean shouldSave = !saveOnlyBest || (valLoss < bestLoss);

        if (!shouldSave) {
            logger.debug("Skipping checkpoint save (not best model)");
            return null;
        }

        if (valLoss < bestLoss) {
            bestLoss = valLoss;
        }

        // Create checkpoint subdirectory
        String timestamp = LocalDateTime.now()
            .format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String checkpointName = String.format("checkpoint_epoch%03d_%s", epoch, timestamp);
        String checkpointPath = Paths.get(checkpointDir, checkpointName).toString();

        try {
            Files.createDirectories(Paths.get(checkpointPath));

            // Save model weights
            String weightsPath = Paths.get(checkpointPath, MODEL_WEIGHTS_FILE).toString();
            model.save(weightsPath);
            logger.debug("Saved model weights to: {}", weightsPath);

            // Save configuration
            String configPath = Paths.get(checkpointPath, CONFIG_FILE).toString();
            config.saveToFile(configPath);
            logger.debug("Saved config to: {}", configPath);

            // Save metadata
            saveMetadata(checkpointPath, epoch, valLoss, model, config);

            // Save metrics
            if (metrics != null) {
                String metricsPath = Paths.get(checkpointPath, METRICS_FILE).toString();
                metrics.saveToCSV(metricsPath);
                logger.debug("Saved metrics to: {}", metricsPath);
            }

            checkpointCount++;

            logger.info("Checkpoint saved: epoch={}, val_loss={:.6f}, path={}",
                       epoch, valLoss, checkpointPath);

            // Clean up old checkpoints if needed
            if (keepLast > 0) {
                cleanupOldCheckpoints();
            }

            return checkpointPath;

        } catch (Exception e) {
            logger.error("Failed to save checkpoint", e);
            throw new RuntimeException("Checkpoint save failed", e);
        }
    }

    /**
     * Saves metadata to JSON file.
     */
    private void saveMetadata(String checkpointPath, int epoch, double valLoss,
                             VariationalAutoencoder3D model, VAEConfig config) throws IOException {

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("epoch", epoch);
        metadata.put("val_loss", valLoss);
        metadata.put("timestamp", LocalDateTime.now().toString());
        metadata.put("model_class", model.getClass().getSimpleName());
        metadata.put("latent_dim", config.getLatentDim());
        metadata.put("input_size", config.getInputSize());
        metadata.put("architecture", config.getArchitectureType().toString());
        metadata.put("checkpoint_count", checkpointCount);

        String metadataPath = Paths.get(checkpointPath, METADATA_FILE).toString();
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        try (FileWriter writer = new FileWriter(metadataPath)) {
            gson.toJson(metadata, writer);
        }

        logger.debug("Saved metadata to: {}", metadataPath);
    }

    /**
     * Loads a model from checkpoint.
     *
     * @param checkpointPath Path to checkpoint directory
     * @return Loaded VAE model
     */
    public VariationalAutoencoder3D load(String checkpointPath) {
        try {
            // Load configuration
            String configPath = Paths.get(checkpointPath, CONFIG_FILE).toString();
            VAEConfig config = VAEConfig.loadFromFile(configPath);
            logger.debug("Loaded config from: {}", configPath);

            // Create model
            VariationalAutoencoder3D model = new VariationalAutoencoder3D(config);

            // Load weights
            String weightsPath = Paths.get(checkpointPath, MODEL_WEIGHTS_FILE).toString();
            model.load(weightsPath);
            logger.info("Model loaded from checkpoint: {}", checkpointPath);

            return model;

        } catch (Exception e) {
            logger.error("Failed to load checkpoint from: {}", checkpointPath, e);
            throw new RuntimeException("Checkpoint load failed", e);
        }
    }

    /**
     * Loads metadata from checkpoint.
     *
     * @param checkpointPath Path to checkpoint
     * @return Metadata map
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> loadMetadata(String checkpointPath) {
        try {
            String metadataPath = Paths.get(checkpointPath, METADATA_FILE).toString();
            Gson gson = new Gson();

            try (FileReader reader = new FileReader(metadataPath)) {
                return gson.fromJson(reader, Map.class);
            }

        } catch (IOException e) {
            logger.error("Failed to load metadata from: {}", checkpointPath, e);
            return new HashMap<>();
        }
    }

    /**
     * Finds the latest checkpoint in directory.
     *
     * @return Path to latest checkpoint, or null if none found
     */
    public String findLatestCheckpoint() {
        try {
            File dir = new File(checkpointDir);
            if (!dir.exists() || !dir.isDirectory()) {
                return null;
            }

            File[] checkpoints = dir.listFiles(File::isDirectory);
            if (checkpoints == null || checkpoints.length == 0) {
                return null;
            }

            // Find most recent by modification time
            File latest = null;
            long latestTime = 0;

            for (File checkpoint : checkpoints) {
                long modTime = checkpoint.lastModified();
                if (modTime > latestTime) {
                    latestTime = modTime;
                    latest = checkpoint;
                }
            }

            return latest != null ? latest.getAbsolutePath() : null;

        } catch (Exception e) {
            logger.error("Error finding latest checkpoint", e);
            return null;
        }
    }

    /**
     * Cleans up old checkpoints, keeping only the most recent.
     */
    private void cleanupOldCheckpoints() {
        try {
            File dir = new File(checkpointDir);
            File[] checkpoints = dir.listFiles(File::isDirectory);

            if (checkpoints == null || checkpoints.length <= keepLast) {
                return; // Nothing to clean up
            }

            // Sort by modification time
            java.util.Arrays.sort(checkpoints,
                (f1, f2) -> Long.compare(f2.lastModified(), f1.lastModified()));

            // Delete old checkpoints
            for (int i = keepLast; i < checkpoints.length; i++) {
                deleteDirectory(checkpoints[i]);
                logger.debug("Deleted old checkpoint: {}", checkpoints[i].getName());
            }

        } catch (Exception e) {
            logger.warn("Failed to cleanup old checkpoints", e);
        }
    }

    /**
     * Recursively deletes a directory.
     */
    private void deleteDirectory(File dir) {
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    deleteDirectory(file);
                } else {
                    file.delete();
                }
            }
        }
        dir.delete();
    }

    /**
     * Gets checkpoint directory.
     *
     * @return Checkpoint directory path
     */
    public String getCheckpointDir() {
        return checkpointDir;
    }

    /**
     * Gets best validation loss seen.
     *
     * @return Best validation loss
     */
    public double getBestLoss() {
        return bestLoss;
    }

    /**
     * Gets total number of checkpoints saved.
     *
     * @return Checkpoint count
     */
    public int getCheckpointCount() {
        return checkpointCount;
    }
}
