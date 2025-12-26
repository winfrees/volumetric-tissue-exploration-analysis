package vtea.deeplearning.models;

import org.bytedeco.pytorch.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 3D Variational Autoencoder (VAE) for volumetric data.
 *
 * <p>This is the main VAE model that combines an encoder and decoder
 * to learn a probabilistic latent representation of 3D cellular images.</p>
 *
 * <p>The VAE learns to:</p>
 * <ul>
 *   <li>Encode 3D volumes into a low-dimensional latent space</li>
 *   <li>Sample from the latent distribution using the reparameterization trick</li>
 *   <li>Decode latent vectors back to reconstructed volumes</li>
 *   <li>Provide a smooth, continuous latent manifold for exploration</li>
 * </ul>
 *
 * <p>Architecture:</p>
 * <pre>
 * Input x [B, C, D, H, W]
 *   ↓
 * Encoder → μ, log σ²
 *   ↓
 * Reparameterization: z = μ + σ ⊙ ε  (ε ~ N(0,I))
 *   ↓
 * Decoder → Reconstruction x̂
 *
 * Loss = Reconstruction Loss + β × KL Divergence
 * </pre>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class VariationalAutoencoder3D extends AbstractDeepLearningModel {

    private static final Logger logger = LoggerFactory.getLogger(VariationalAutoencoder3D.class);

    private final VAEEncoder3D encoder;
    private final VAEDecoder3D decoder;
    private final VAEConfig config;
    private final int latentDim;

    /**
     * Creates a 3D VAE with the specified configuration.
     *
     * @param config VAE configuration
     */
    public VariationalAutoencoder3D(VAEConfig config) {
        super(config.isUseGPU());

        if (config == null) {
            throw new IllegalArgumentException("Config cannot be null");
        }

        // Validate configuration
        config.validate();

        this.config = config;
        this.latentDim = config.getLatentDim();

        logger.info("Creating VAE with config: {}", config);

        // Initialize encoder
        this.encoder = new VAEEncoder3D(
            config.getNumChannels(),
            config.getLatentDim(),
            config.getEncoderChannels()
        );

        // Initialize decoder (mirrors encoder)
        this.decoder = new VAEDecoder3D(
            config.getLatentDim(),
            config.getNumChannels(),
            config.getEncoderChannels(),
            config.getInputSize()
        );

        // Register submodules
        register_module("encoder", encoder);
        register_module("decoder", decoder);

        logger.info("VAE created successfully: latent_dim={}", latentDim);
    }

    /**
     * Forward pass through the complete VAE.
     *
     * <p>Performs encoding, sampling via reparameterization, and decoding.</p>
     *
     * @param x Input tensor [B, C, D, H, W]
     * @return VAEOutput containing reconstruction, μ, log σ², and sampled z
     */
    @Override
    public VAEOutput forward(Tensor x) {
        // Validate input
        validateInputDimensions(x, 5); // 5D tensor [B, C, D, H, W]
        x = ensureCorrectDevice(x);

        logger.debug("VAE forward: input shape = {}",
                    java.util.Arrays.toString(x.sizes()));

        // Encode to latent distribution parameters
        VAEEncoder3D.EncoderOutput encoded = encoder.forward(x);
        Tensor mu = encoded.mu;
        Tensor logVar = encoded.logVar;

        logger.debug("Encoded: mu shape = {}, logVar shape = {}",
                    java.util.Arrays.toString(mu.sizes()),
                    java.util.Arrays.toString(logVar.sizes()));

        // Reparameterization trick
        Tensor z = reparameterize(mu, logVar);

        logger.debug("Sampled z: shape = {}", java.util.Arrays.toString(z.sizes()));

        // Decode to reconstruction
        Tensor reconstruction = decoder.forward(z);

        logger.debug("Reconstruction: shape = {}",
                    java.util.Arrays.toString(reconstruction.sizes()));

        return new VAEOutput(reconstruction, mu, logVar, z);
    }

    /**
     * Reparameterization trick for sampling from latent distribution.
     *
     * <p>Computes: z = μ + σ ⊙ ε, where ε ~ N(0, I)</p>
     *
     * <p>This allows gradients to flow through the sampling operation
     * during backpropagation.</p>
     *
     * @param mu Mean of latent distribution [B, latentDim]
     * @param logVar Log variance of latent distribution [B, latentDim]
     * @return Sampled latent vector z [B, latentDim]
     */
    private Tensor reparameterize(Tensor mu, Tensor logVar) {
        if (!isTraining) {
            // During inference, just use the mean (deterministic)
            logger.debug("Inference mode: using mean without sampling");
            return mu;
        }

        // Compute standard deviation: σ = exp(0.5 × log(σ²))
        Tensor std = logVar.mul(0.5).exp();

        // Sample epsilon from standard normal distribution
        Tensor eps = randn_like(std);

        // Reparameterization: z = μ + σ ⊙ ε
        Tensor z = mu.add(std.mul(eps));

        return z;
    }

    /**
     * Encodes input volumes to latent distribution parameters.
     *
     * @param x Input tensor [B, C, D, H, W]
     * @return EncoderOutput with μ and log σ²
     */
    public VAEEncoder3D.EncoderOutput encode(Tensor x) {
        validateInputDimensions(x, 5);
        x = ensureCorrectDevice(x);

        return encoder.forward(x);
    }

    /**
     * Decodes latent vectors to reconstructed volumes.
     *
     * @param z Latent vectors [B, latentDim]
     * @return Reconstructed volumes [B, C, D, H, W]
     */
    public Tensor decode(Tensor z) {
        if (z.sizes().length != 2) {
            throw new IllegalArgumentException("Latent tensor must be 2D [B, latentDim]");
        }
        z = ensureCorrectDevice(z);

        return decoder.forward(z);
    }

    /**
     * Generates new samples by sampling from the prior distribution N(0, I).
     *
     * @param numSamples Number of samples to generate
     * @return Generated volumes [numSamples, C, D, H, W]
     */
    public Tensor sample(int numSamples) {
        if (numSamples <= 0) {
            throw new IllegalArgumentException("Number of samples must be positive");
        }

        logger.info("Generating {} samples from prior", numSamples);

        // Sample from prior N(0, I)
        Tensor z = randn(new long[]{numSamples, latentDim});

        if (useGPU) {
            z = z.cuda();
        }

        // Decode to images
        return decoder.forward(z);
    }

    /**
     * Linearly interpolates between two images in latent space.
     *
     * @param x1 First input image [1, C, D, H, W]
     * @param x2 Second input image [1, C, D, H, W]
     * @param steps Number of interpolation steps (including endpoints)
     * @return Array of interpolated images
     */
    public Tensor[] interpolate(Tensor x1, Tensor x2, int steps) {
        if (steps < 2) {
            throw new IllegalArgumentException("Steps must be at least 2");
        }

        logger.info("Interpolating between two images with {} steps", steps);

        // Encode both images (use mean, not sample)
        boolean wasTraining = isTraining;
        eval(); // Use deterministic encoding

        Tensor mu1 = encoder.forward(x1).mu;
        Tensor mu2 = encoder.forward(x2).mu;

        if (wasTraining) {
            train();
        }

        Tensor[] interpolations = new Tensor[steps];

        // Linearly interpolate in latent space
        for (int i = 0; i < steps; i++) {
            double alpha = (double) i / (steps - 1);

            // z = (1 - α) × mu1 + α × mu2
            Tensor z = mu1.mul(1.0 - alpha).add(mu2.mul(alpha));

            // Decode interpolated latent vector
            interpolations[i] = decoder.forward(z);

            logger.debug("Interpolation step {}/{}: alpha = {:.3f}",
                        i + 1, steps, alpha);
        }

        return interpolations;
    }

    /**
     * Computes reconstruction for a single input.
     *
     * <p>Uses mean of latent distribution (deterministic) for stable reconstructions.</p>
     *
     * @param x Input volume [1, C, D, H, W]
     * @return Reconstructed volume [1, C, D, H, W]
     */
    public Tensor reconstruct(Tensor x) {
        boolean wasTraining = isTraining;
        eval(); // Deterministic reconstruction

        // Encode and use mean
        Tensor mu = encoder.forward(x).mu;

        // Decode
        Tensor reconstruction = decoder.forward(mu);

        if (wasTraining) {
            train();
        }

        return reconstruction;
    }

    /**
     * Gets the latent dimensionality.
     *
     * @return Latent space dimensionality
     */
    public int getLatentDim() {
        return latentDim;
    }

    /**
     * Gets the VAE configuration.
     *
     * @return Configuration object
     */
    public VAEConfig getConfig() {
        return config;
    }

    /**
     * Gets the encoder.
     *
     * @return Encoder module
     */
    public VAEEncoder3D getEncoder() {
        return encoder;
    }

    /**
     * Gets the decoder.
     *
     * @return Decoder module
     */
    public VAEDecoder3D getDecoder() {
        return decoder;
    }

    @Override
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Variational Autoencoder 3D ===\n");
        sb.append(super.getSummary());
        sb.append("\nConfiguration:\n");
        sb.append("  Input size: ").append(config.getInputSize()).append("³\n");
        sb.append("  Channels: ").append(config.getNumChannels()).append("\n");
        sb.append("  Latent dim: ").append(latentDim).append("\n");
        sb.append("  Architecture: ").append(config.getArchitectureType()).append("\n");
        sb.append("\nEncoder:\n  ").append(encoder.toString()).append("\n");
        sb.append("\nDecoder:\n  ").append(decoder.toString()).append("\n");

        return sb.toString();
    }

    /**
     * Output container for VAE forward pass.
     *
     * <p>Contains all relevant tensors produced during the forward pass,
     * needed for computing the VAE loss.</p>
     */
    public static class VAEOutput {
        /** Reconstructed volume [B, C, D, H, W] */
        public final Tensor reconstruction;

        /** Mean of latent distribution [B, latentDim] */
        public final Tensor mu;

        /** Log variance of latent distribution [B, latentDim] */
        public final Tensor logVar;

        /** Sampled latent vector [B, latentDim] */
        public final Tensor z;

        /**
         * Creates VAE output.
         *
         * @param reconstruction Reconstructed volume
         * @param mu Latent mean
         * @param logVar Latent log variance
         * @param z Sampled latent vector
         */
        public VAEOutput(Tensor reconstruction, Tensor mu, Tensor logVar, Tensor z) {
            this.reconstruction = reconstruction;
            this.mu = mu;
            this.logVar = logVar;
            this.z = z;
        }

        /**
         * Computes standard deviation from log variance.
         *
         * @return Standard deviation [B, latentDim]
         */
        public Tensor getStd() {
            return logVar.mul(0.5).exp();
        }

        /**
         * Computes variance from log variance.
         *
         * @return Variance [B, latentDim]
         */
        public Tensor getVar() {
            return logVar.exp();
        }
    }

    @Override
    public String toString() {
        return String.format("VariationalAutoencoder3D(latent_dim=%d, architecture=%s)",
                           latentDim, config.getArchitectureType());
    }
}
