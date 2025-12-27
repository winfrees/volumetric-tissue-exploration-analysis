package vtea.deeplearning.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Tracks and records training metrics for VAE training.
 *
 * <p>Follows VTEA patterns for progress tracking and logging,
 * compatible with VTEA's ProgressListener interface.</p>
 *
 * <p>Metrics tracked:</p>
 * <ul>
 *   <li>Total loss per epoch (train and validation)</li>
 *   <li>Reconstruction loss</li>
 *   <li>KL divergence</li>
 *   <li>ELBO (Evidence Lower Bound)</li>
 *   <li>Best model tracking</li>
 *   <li>Early stopping patience</li>
 * </ul>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class TrainingMetrics {

    private static final Logger logger = LoggerFactory.getLogger(TrainingMetrics.class);

    // Training history
    private final List<EpochMetrics> trainHistory;
    private final List<EpochMetrics> valHistory;

    // Current epoch metrics
    private double epochTotalLoss;
    private double epochReconLoss;
    private double epochKLLoss;
    private double epochELBO;
    private int batchCount;

    // Best model tracking
    private double bestValLoss;
    private int bestEpoch;

    // Early stopping
    private final int patience;
    private int patienceCounter;
    private boolean shouldStop;

    /**
     * Creates a TrainingMetrics tracker.
     *
     * @param patience Number of epochs without improvement before early stopping
     */
    public TrainingMetrics(int patience) {
        this.trainHistory = new ArrayList<>();
        this.valHistory = new ArrayList<>();
        this.patience = patience;
        this.patienceCounter = 0;
        this.bestValLoss = Double.MAX_VALUE;
        this.bestEpoch = -1;
        this.shouldStop = false;

        resetEpoch();

        logger.info("TrainingMetrics initialized with patience={}", patience);
    }

    /**
     * Resets epoch-level accumulators.
     */
    public void resetEpoch() {
        epochTotalLoss = 0.0;
        epochReconLoss = 0.0;
        epochKLLoss = 0.0;
        epochELBO = 0.0;
        batchCount = 0;
    }

    /**
     * Updates metrics with a batch result.
     *
     * @param totalLoss Total loss for batch
     * @param reconLoss Reconstruction loss
     * @param klLoss KL divergence
     * @param elbo ELBO value
     */
    public void updateBatch(double totalLoss, double reconLoss,
                           double klLoss, double elbo) {
        epochTotalLoss += totalLoss;
        epochReconLoss += reconLoss;
        epochKLLoss += klLoss;
        epochELBO += elbo;
        batchCount++;
    }

    /**
     * Finalizes epoch and stores averaged metrics.
     *
     * @param isValidation Whether these are validation metrics
     * @return Averaged epoch metrics
     */
    public EpochMetrics finalizeEpoch(boolean isValidation) {
        if (batchCount == 0) {
            logger.warn("Finalizing epoch with no batches processed");
            return new EpochMetrics(0, 0, 0, 0, 0);
        }

        double avgTotal = epochTotalLoss / batchCount;
        double avgRecon = epochReconLoss / batchCount;
        double avgKL = epochKLLoss / batchCount;
        double avgELBO = epochELBO / batchCount;

        EpochMetrics metrics = new EpochMetrics(
            trainHistory.size() + valHistory.size(),
            avgTotal, avgRecon, avgKL, avgELBO
        );

        if (isValidation) {
            valHistory.add(metrics);
            updateBestModel(avgTotal);
        } else {
            trainHistory.add(metrics);
        }

        resetEpoch();

        return metrics;
    }

    /**
     * Updates best model tracking and early stopping.
     *
     * @param valLoss Validation loss
     */
    private void updateBestModel(double valLoss) {
        if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            bestEpoch = valHistory.size() - 1;
            patienceCounter = 0;

            logger.info("New best model! Validation loss: {:.6f} at epoch {}",
                       bestValLoss, bestEpoch);
        } else {
            patienceCounter++;

            if (patienceCounter >= patience) {
                shouldStop = true;
                logger.info("Early stopping triggered after {} epochs without improvement",
                           patience);
            } else {
                logger.debug("No improvement for {} epochs (patience: {}/{})",
                            patienceCounter, patienceCounter, patience);
            }
        }
    }

    /**
     * Checks if training should stop early.
     *
     * @return true if early stopping triggered
     */
    public boolean shouldStop() {
        return shouldStop;
    }

    /**
     * Gets the best validation loss.
     *
     * @return Best validation loss
     */
    public double getBestValLoss() {
        return bestValLoss;
    }

    /**
     * Gets the epoch with best validation loss.
     *
     * @return Best epoch index
     */
    public int getBestEpoch() {
        return bestEpoch;
    }

    /**
     * Gets training history.
     *
     * @return List of training epoch metrics
     */
    public List<EpochMetrics> getTrainHistory() {
        return new ArrayList<>(trainHistory);
    }

    /**
     * Gets validation history.
     *
     * @return List of validation epoch metrics
     */
    public List<EpochMetrics> getValHistory() {
        return new ArrayList<>(valHistory);
    }

    /**
     * Saves metrics to CSV file.
     *
     * @param filepath Path to save CSV
     * @throws IOException if save fails
     */
    public void saveToCSV(String filepath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filepath))) {
            // Header
            writer.println("epoch,split,total_loss,recon_loss,kl_loss,elbo");

            // Training metrics
            for (EpochMetrics metrics : trainHistory) {
                writer.printf("%d,train,%.6f,%.6f,%.6f,%.6f%n",
                            metrics.epoch,
                            metrics.totalLoss,
                            metrics.reconLoss,
                            metrics.klLoss,
                            metrics.elbo);
            }

            // Validation metrics
            for (EpochMetrics metrics : valHistory) {
                writer.printf("%d,val,%.6f,%.6f,%.6f,%.6f%n",
                            metrics.epoch,
                            metrics.totalLoss,
                            metrics.reconLoss,
                            metrics.klLoss,
                            metrics.elbo);
            }
        }

        logger.info("Metrics saved to: {}", filepath);
    }

    /**
     * Gets summary statistics.
     *
     * @return Summary string
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Training Metrics Summary ===\n");
        sb.append(String.format("Total epochs: %d\n", trainHistory.size()));
        sb.append(String.format("Best epoch: %d\n", bestEpoch));
        sb.append(String.format("Best val loss: %.6f\n", bestValLoss));

        if (!trainHistory.isEmpty()) {
            EpochMetrics first = trainHistory.get(0);
            EpochMetrics last = trainHistory.get(trainHistory.size() - 1);
            sb.append(String.format("Initial train loss: %.6f\n", first.totalLoss));
            sb.append(String.format("Final train loss: %.6f\n", last.totalLoss));
            sb.append(String.format("Improvement: %.2f%%\n",
                     (first.totalLoss - last.totalLoss) / first.totalLoss * 100));
        }

        if (shouldStop) {
            sb.append("Early stopping triggered\n");
        }

        return sb.toString();
    }

    /**
     * Container for metrics from a single epoch.
     */
    public static class EpochMetrics {
        public final int epoch;
        public final double totalLoss;
        public final double reconLoss;
        public final double klLoss;
        public final double elbo;

        public EpochMetrics(int epoch, double totalLoss,
                           double reconLoss, double klLoss, double elbo) {
            this.epoch = epoch;
            this.totalLoss = totalLoss;
            this.reconLoss = reconLoss;
            this.klLoss = klLoss;
            this.elbo = elbo;
        }

        @Override
        public String toString() {
            return String.format("Epoch %d: Loss=%.4f (Recon=%.4f, KL=%.4f, ELBO=%.4f)",
                               epoch, totalLoss, reconLoss, klLoss, elbo);
        }
    }

    /**
     * Computes moving average of last N epochs.
     *
     * @param n Number of epochs to average
     * @param isValidation Whether to use validation history
     * @return Moving average of total loss
     */
    public double getMovingAverage(int n, boolean isValidation) {
        List<EpochMetrics> history = isValidation ? valHistory : trainHistory;

        if (history.isEmpty()) {
            return 0.0;
        }

        int start = Math.max(0, history.size() - n);
        double sum = 0.0;

        for (int i = start; i < history.size(); i++) {
            sum += history.get(i).totalLoss;
        }

        return sum / (history.size() - start);
    }

    /**
     * Checks if training is improving.
     *
     * @param lookback Number of epochs to check
     * @return true if loss is decreasing
     */
    public boolean isImproving(int lookback) {
        if (trainHistory.size() < lookback + 1) {
            return true; // Not enough data
        }

        double recent = getMovingAverage(lookback / 2, false);
        double older = getMovingAverage(lookback, false);

        return recent < older;
    }
}
