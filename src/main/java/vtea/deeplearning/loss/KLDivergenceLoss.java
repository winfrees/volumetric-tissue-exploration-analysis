package vtea.deeplearning.loss;

import org.bytedeco.pytorch.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * KL Divergence loss for Variational Autoencoder.
 *
 * <p>Computes the Kullback-Leibler divergence between the learned latent
 * distribution q(z|x) ~ N(μ, σ²) and the prior p(z) ~ N(0, I).</p>
 *
 * <p>Analytical formula for Gaussian distributions:</p>
 * <pre>
 * KL(N(μ, σ²) || N(0, I)) = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
 * </pre>
 *
 * <p>This regularizes the latent space to be close to a standard normal
 * distribution, enabling smooth interpolation and sampling.</p>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class KLDivergenceLoss {

    private static final Logger logger = LoggerFactory.getLogger(KLDivergenceLoss.class);

    /**
     * Computes KL divergence loss (mean over batch).
     *
     * @param mu Mean of latent distribution [B, latentDim]
     * @param logVar Log variance of latent distribution [B, latentDim]
     * @return Scalar KL divergence loss (mean over batch)
     */
    public Tensor compute(Tensor mu, Tensor logVar) {
        if (mu == null || logVar == null) {
            throw new IllegalArgumentException("mu and logVar cannot be null");
        }

        // Validate shapes match
        long[] muShape = mu.sizes();
        long[] logVarShape = logVar.sizes();

        if (muShape.length != 2 || logVarShape.length != 2) {
            throw new IllegalArgumentException("mu and logVar must be 2D tensors [B, latentDim]");
        }

        if (muShape[0] != logVarShape[0] || muShape[1] != logVarShape[1]) {
            throw new IllegalArgumentException(
                String.format("Shape mismatch: mu=%s, logVar=%s",
                            java.util.Arrays.toString(muShape),
                            java.util.Arrays.toString(logVarShape)));
        }

        // KL divergence formula:
        // KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        //
        // Breaking it down:
        // 1. 1 + log_var: normalizes for variance
        // 2. - mu^2: penalizes mean far from 0
        // 3. - exp(log_var): penalizes variance far from 1

        Tensor kl = logVar.add(1.0)           // 1 + log(σ²)
                         .sub(mu.pow(2.0))    // - μ²
                         .sub(logVar.exp());  // - σ²

        // Sum over latent dimensions, mean over batch
        kl = kl.sum(new long[]{1})    // Sum over latent dimensions
               .mul(-0.5)              // Multiply by -0.5
               .mean();                // Mean over batch

        logger.debug("KL divergence computed: {}", kl.item().doubleValue());

        return kl;
    }

    /**
     * Computes per-sample KL divergence (for analysis/debugging).
     *
     * <p>Returns KL divergence for each sample in the batch,
     * useful for identifying problematic samples.</p>
     *
     * @param mu Mean of latent distribution [B, latentDim]
     * @param logVar Log variance of latent distribution [B, latentDim]
     * @return Per-sample KL divergence [B]
     */
    public Tensor computePerSample(Tensor mu, Tensor logVar) {
        if (mu == null || logVar == null) {
            throw new IllegalArgumentException("mu and logVar cannot be null");
        }

        Tensor kl = logVar.add(1.0)
                         .sub(mu.pow(2.0))
                         .sub(logVar.exp());

        // Sum over latent dimensions only (keep batch dimension)
        kl = kl.sum(new long[]{1}).mul(-0.5);

        return kl;
    }

    /**
     * Computes per-dimension KL divergence (for disentanglement analysis).
     *
     * <p>Returns KL divergence for each latent dimension averaged over batch,
     * useful for analyzing which dimensions are being utilized.</p>
     *
     * @param mu Mean of latent distribution [B, latentDim]
     * @param logVar Log variance of latent distribution [B, latentDim]
     * @return Per-dimension KL divergence [latentDim]
     */
    public Tensor computePerDimension(Tensor mu, Tensor logVar) {
        if (mu == null || logVar == null) {
            throw new IllegalArgumentException("mu and logVar cannot be null");
        }

        Tensor kl = logVar.add(1.0)
                         .sub(mu.pow(2.0))
                         .sub(logVar.exp());

        kl = kl.mul(-0.5);

        // Mean over batch (keep latent dimension)
        kl = kl.mean(new long[]{0});

        return kl;
    }

    /**
     * Computes KL divergence with free bits constraint.
     *
     * <p>Free bits allows a minimum KL per dimension, preventing posterior collapse.
     * Each latent dimension is guaranteed at least 'freeBits' nats of information.</p>
     *
     * @param mu Mean of latent distribution [B, latentDim]
     * @param logVar Log variance of latent distribution [B, latentDim]
     * @param freeBits Minimum KL per dimension (e.g., 0.5 nats)
     * @return KL divergence with free bits constraint
     */
    public Tensor computeWithFreeBits(Tensor mu, Tensor logVar, double freeBits) {
        if (freeBits < 0) {
            throw new IllegalArgumentException("freeBits must be non-negative");
        }

        // Compute per-dimension KL (mean over batch)
        Tensor klPerDim = computePerDimension(mu, logVar);

        // Clamp to minimum free bits
        klPerDim = klPerDim.clamp_min(freeBits);

        // Sum over dimensions
        Tensor kl = klPerDim.sum();

        logger.debug("KL divergence with free bits ({}): {}",
                    freeBits, kl.item().doubleValue());

        return kl;
    }

    /**
     * Checks if posterior collapse has occurred.
     *
     * <p>Posterior collapse happens when KL → 0, meaning the encoder
     * ignores the input and the model becomes a deterministic autoencoder.</p>
     *
     * @param mu Mean of latent distribution
     * @param logVar Log variance of latent distribution
     * @param threshold KL threshold below which collapse is detected (e.g., 0.01)
     * @return true if posterior collapse detected
     */
    public boolean detectPosteriorCollapse(Tensor mu, Tensor logVar, double threshold) {
        Tensor kl = compute(mu, logVar);
        double klValue = kl.item().doubleValue();

        boolean collapsed = klValue < threshold;

        if (collapsed) {
            logger.warn("Posterior collapse detected! KL = {} < threshold {}",
                       klValue, threshold);
        }

        return collapsed;
    }
}
