package vtea.deeplearning.loss;

import org.bytedeco.pytorch.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import vtea.deeplearning.models.VAEConfig;

/**
 * Combined loss function for Variational Autoencoder.
 *
 * <p>Combines reconstruction loss and KL divergence with configurable weighting:</p>
 * <pre>
 * Total Loss = Reconstruction Loss + β × KL Divergence
 * </pre>
 *
 * <p>Features:</p>
 * <ul>
 *   <li>β-VAE: Adjustable KL weighting for disentanglement</li>
 *   <li>KL warmup: Gradually increase KL weight to prevent collapse</li>
 *   <li>Per-component tracking for monitoring</li>
 *   <li>Automatic ELBO (Evidence Lower Bound) computation</li>
 * </ul>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class VAELoss {

    private static final Logger logger = LoggerFactory.getLogger(VAELoss.class);

    private final ReconstructionLoss reconstructionLoss;
    private final KLDivergenceLoss klLoss;
    private final double beta;
    private final boolean useKLWarmup;
    private final int warmupEpochs;

    private int currentEpoch;

    /**
     * Creates a VAE loss function.
     *
     * @param reconstructionType Type of reconstruction loss (MSE, BCE, L1)
     * @param beta β parameter for weighting KL divergence (β-VAE)
     * @param useKLWarmup Whether to use KL warmup
     * @param warmupEpochs Number of epochs for warmup (if enabled)
     */
    public VAELoss(VAEConfig.ReconstructionType reconstructionType,
                   double beta,
                   boolean useKLWarmup,
                   int warmupEpochs) {

        if (reconstructionType == null) {
            throw new IllegalArgumentException("Reconstruction type cannot be null");
        }
        if (beta < 0) {
            throw new IllegalArgumentException("Beta must be non-negative");
        }
        if (warmupEpochs < 0) {
            throw new IllegalArgumentException("Warmup epochs must be non-negative");
        }

        this.reconstructionLoss = new ReconstructionLoss(reconstructionType);
        this.klLoss = new KLDivergenceLoss();
        this.beta = beta;
        this.useKLWarmup = useKLWarmup;
        this.warmupEpochs = warmupEpochs;
        this.currentEpoch = 0;

        logger.info("VAELoss initialized: recon_type={}, beta={}, warmup={}, warmup_epochs={}",
                   reconstructionType, beta, useKLWarmup, warmupEpochs);
    }

    /**
     * Creates a VAE loss from configuration.
     *
     * @param config VAE configuration
     */
    public VAELoss(VAEConfig config) {
        this(config.getReconstructionType(),
             config.getBeta(),
             config.isUseKLWarmup(),
             config.getWarmupEpochs());
    }

    /**
     * Computes the complete VAE loss.
     *
     * @param reconstruction Reconstructed tensor [B, C, D, H, W]
     * @param target Original input tensor [B, C, D, H, W]
     * @param mu Latent mean [B, latentDim]
     * @param logVar Latent log variance [B, latentDim]
     * @return LossOutput containing all loss components
     */
    public LossOutput compute(Tensor reconstruction, Tensor target,
                             Tensor mu, Tensor logVar) {

        if (reconstruction == null || target == null || mu == null || logVar == null) {
            throw new IllegalArgumentException("Loss inputs cannot be null");
        }

        // Compute reconstruction loss
        Tensor reconLoss = reconstructionLoss.compute(reconstruction, target);

        // Compute KL divergence
        Tensor kl = klLoss.compute(mu, logVar);

        // Get current KL weight (with warmup)
        double klWeight = getKLWeight();

        // Total loss: reconstruction + β × KL
        Tensor totalLoss = reconLoss.add(kl.mul(klWeight));

        // Compute ELBO (Evidence Lower Bound)
        // ELBO = -Loss = -(Reconstruction + KL)
        // Higher ELBO is better (closer to log p(x))
        double elbo = -totalLoss.item().doubleValue();

        logger.debug("Loss computed: total={:.4f}, recon={:.4f}, kl={:.4f}, " +
                    "kl_weight={:.3f}, elbo={:.4f}",
                    totalLoss.item().doubleValue(),
                    reconLoss.item().doubleValue(),
                    kl.item().doubleValue(),
                    klWeight,
                    elbo);

        return new LossOutput(totalLoss, reconLoss, kl, klWeight, elbo);
    }

    /**
     * Computes per-sample losses for analysis.
     *
     * @param reconstruction Reconstructed tensor
     * @param target Original tensor
     * @param mu Latent mean
     * @param logVar Latent log variance
     * @return Per-sample total loss [B]
     */
    public Tensor computePerSample(Tensor reconstruction, Tensor target,
                                   Tensor mu, Tensor logVar) {

        // Per-sample reconstruction loss
        Tensor reconLossPerSample = reconstructionLoss.computePerSample(reconstruction, target);

        // Per-sample KL divergence
        Tensor klPerSample = klLoss.computePerSample(mu, logVar);

        // Combine with current KL weight
        double klWeight = getKLWeight();
        Tensor totalLossPerSample = reconLossPerSample.add(klPerSample.mul(klWeight));

        return totalLossPerSample;
    }

    /**
     * Gets the current KL divergence weight (handles warmup).
     *
     * <p>During warmup, KL weight increases linearly from 0 to β:</p>
     * <pre>
     * weight = β × (current_epoch / warmup_epochs)
     * </pre>
     *
     * @return Current KL weight
     */
    private double getKLWeight() {
        if (!useKLWarmup) {
            return beta;
        }

        // Linear warmup from 0 to beta
        if (currentEpoch < warmupEpochs) {
            double progress = (double) currentEpoch / warmupEpochs;
            double weight = beta * progress;

            logger.debug("KL warmup: epoch {}/{}, weight = {:.3f}",
                        currentEpoch, warmupEpochs, weight);

            return weight;
        }

        return beta;
    }

    /**
     * Sets the current epoch (for KL warmup).
     *
     * @param epoch Current training epoch (0-indexed)
     */
    public void setEpoch(int epoch) {
        if (epoch < 0) {
            throw new IllegalArgumentException("Epoch must be non-negative");
        }

        this.currentEpoch = epoch;

        if (useKLWarmup && epoch < warmupEpochs) {
            logger.debug("KL warmup active: epoch {}/{}", epoch, warmupEpochs);
        }
    }

    /**
     * Gets the beta parameter.
     *
     * @return Beta value
     */
    public double getBeta() {
        return beta;
    }

    /**
     * Checks if KL warmup is enabled.
     *
     * @return true if using KL warmup
     */
    public boolean isUsingKLWarmup() {
        return useKLWarmup;
    }

    /**
     * Gets the number of warmup epochs.
     *
     * @return Warmup epochs
     */
    public int getWarmupEpochs() {
        return warmupEpochs;
    }

    /**
     * Container for VAE loss components.
     */
    public static class LossOutput {
        /** Total VAE loss (reconstruction + β × KL) */
        public final Tensor totalLoss;

        /** Reconstruction loss component */
        public final Tensor reconstructionLoss;

        /** KL divergence component */
        public final Tensor klDivergence;

        /** Current KL weight (β with optional warmup) */
        public final double klWeight;

        /** Evidence Lower Bound (ELBO = -totalLoss) */
        public final double elbo;

        /**
         * Creates loss output.
         *
         * @param totalLoss Total loss
         * @param reconLoss Reconstruction loss
         * @param kl KL divergence
         * @param klWeight KL weight used
         * @param elbo ELBO value
         */
        public LossOutput(Tensor totalLoss, Tensor reconLoss,
                         Tensor kl, double klWeight, double elbo) {
            this.totalLoss = totalLoss;
            this.reconstructionLoss = reconLoss;
            this.klDivergence = kl;
            this.klWeight = klWeight;
            this.elbo = elbo;
        }

        /**
         * Gets total loss as double.
         *
         * @return Total loss value
         */
        public double getTotalLossValue() {
            return totalLoss.item().doubleValue();
        }

        /**
         * Gets reconstruction loss as double.
         *
         * @return Reconstruction loss value
         */
        public double getReconstructionLossValue() {
            return reconstructionLoss.item().doubleValue();
        }

        /**
         * Gets KL divergence as double.
         *
         * @return KL divergence value
         */
        public double getKLDivergenceValue() {
            return klDivergence.item().doubleValue();
        }

        @Override
        public String toString() {
            return String.format("VAELoss[total=%.4f, recon=%.4f, kl=%.4f, " +
                               "kl_weight=%.3f, elbo=%.4f]",
                               getTotalLossValue(),
                               getReconstructionLossValue(),
                               getKLDivergenceValue(),
                               klWeight,
                               elbo);
        }
    }

    @Override
    public String toString() {
        return String.format("VAELoss(recon=%s, beta=%.2f, warmup=%s, warmup_epochs=%d)",
                           reconstructionLoss.getLossType(), beta,
                           useKLWarmup, warmupEpochs);
    }
}
