package vtea.deeplearning.models;

import org.bytedeco.pytorch.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 3D Convolutional Encoder for Variational Autoencoder.
 *
 * <p>This encoder network progressively downsamples 3D volumetric input
 * through convolutional blocks and outputs latent distribution parameters
 * (μ and log σ²) for the VAE's probabilistic latent space.</p>
 *
 * <p>Architecture:</p>
 * <pre>
 * Input [B, C, D, H, W]
 *   ↓
 * Conv3D Block 1: C → channels[0]
 *   ↓
 * Conv3D Block 2: channels[0] → channels[1]
 *   ↓
 * ...
 *   ↓
 * Conv3D Block N: channels[N-2] → channels[N-1]
 *   ↓
 * Flatten
 *   ↓
 * FC → μ (latentDim)
 * FC → log σ² (latentDim)
 * </pre>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class VAEEncoder3D extends Module {

    private static final Logger logger = LoggerFactory.getLogger(VAEEncoder3D.class);

    private final int inputChannels;
    private final int latentDim;
    private final int[] channels;

    private Sequential convBlocks;
    private Linear fcMu;
    private Linear fcLogVar;

    /**
     * Creates a 3D encoder for VAE.
     *
     * @param inputChannels Number of input channels (e.g., 1 for grayscale)
     * @param latentDim Dimensionality of latent space
     * @param channels Array specifying channel progression (e.g., [32, 64, 128, 256])
     */
    public VAEEncoder3D(int inputChannels, int latentDim, int[] channels) {
        super();

        if (inputChannels <= 0) {
            throw new IllegalArgumentException("Input channels must be positive");
        }
        if (latentDim <= 0) {
            throw new IllegalArgumentException("Latent dimension must be positive");
        }
        if (channels == null || channels.length == 0) {
            throw new IllegalArgumentException("Channels array must not be empty");
        }

        this.inputChannels = inputChannels;
        this.latentDim = latentDim;
        this.channels = channels;

        logger.info("Creating VAE Encoder: input_channels={}, latent_dim={}, channels={}",
                   inputChannels, latentDim, java.util.Arrays.toString(channels));

        buildNetwork();
    }

    /**
     * Builds the encoder network architecture.
     */
    private void buildNetwork() {
        // Build convolutional blocks
        convBlocks = buildConvolutionalBlocks();
        register_module("conv_blocks", convBlocks);

        // Calculate flattened size after convolutions
        // For input size S and N conv blocks with stride 2, output size ≈ S / 2^N
        // We'll compute this dynamically or use a reasonable estimate
        // For now, assume final spatial size is small (e.g., 4³ for 64³ input with 4 blocks)
        int finalChannels = channels[channels.length - 1];

        // This is an approximation - in practice we'd compute exact size
        // by running a forward pass with dummy input
        int estimatedSpatialSize = 4; // Conservative estimate
        int flattenedSize = finalChannels * estimatedSpatialSize * estimatedSpatialSize * estimatedSpatialSize;

        // Fully connected layers for latent parameters
        fcMu = new Linear(flattenedSize, latentDim);
        fcLogVar = new Linear(flattenedSize, latentDim);

        register_module("fc_mu", fcMu);
        register_module("fc_logvar", fcLogVar);

        logger.info("Encoder built: estimated flattened size = {}, latent dim = {}",
                   flattenedSize, latentDim);
    }

    /**
     * Builds sequential convolutional blocks.
     *
     * @return Sequential module containing all conv blocks
     */
    private Sequential buildConvolutionalBlocks() {
        Sequential blocks = new Sequential();

        int currentChannels = inputChannels;

        for (int i = 0; i < channels.length; i++) {
            int outChannels = channels[i];
            boolean downsample = (i < channels.length - 1); // Downsample all but last block

            Sequential block = createConvBlock(currentChannels, outChannels, downsample);
            blocks.add(block);

            currentChannels = outChannels;

            logger.debug("Added conv block {}: {} → {} channels, downsample={}",
                        i, currentChannels, outChannels, downsample);
        }

        return blocks;
    }

    /**
     * Creates a single convolutional block.
     *
     * <p>Each block contains:</p>
     * <ul>
     *   <li>Conv3d (with optional stride=2 for downsampling)</li>
     *   <li>BatchNorm3d</li>
     *   <li>LeakyReLU</li>
     *   <li>Conv3d (stride=1)</li>
     *   <li>BatchNorm3d</li>
     *   <li>LeakyReLU</li>
     * </ul>
     *
     * @param inChannels Input channels
     * @param outChannels Output channels
     * @param downsample Whether to downsample (stride=2 on first conv)
     * @return Sequential block
     */
    private Sequential createConvBlock(int inChannels, int outChannels, boolean downsample) {
        Sequential block = new Sequential();

        int stride = downsample ? 2 : 1;
        long[] kernelSize = {3, 3, 3};
        long[] padding = {1, 1, 1};

        // First convolution (may downsample)
        Conv3dOptions conv1Options = new Conv3dOptions(inChannels, outChannels, kernelSize);
        conv1Options.stride(new long[]{stride, stride, stride});
        conv1Options.padding(padding);
        Conv3d conv1 = new Conv3d(conv1Options);
        block.add(conv1);

        // Batch normalization
        BatchNorm3d bn1 = new BatchNorm3d(outChannels);
        block.add(bn1);

        // Leaky ReLU activation
        LeakyReLUOptions relu1Options = new LeakyReLUOptions();
        relu1Options.negative_slope(0.2);
        LeakyReLU relu1 = new LeakyReLU(relu1Options);
        block.add(relu1);

        // Second convolution (no downsampling)
        Conv3dOptions conv2Options = new Conv3dOptions(outChannels, outChannels, kernelSize);
        conv2Options.stride(new long[]{1, 1, 1});
        conv2Options.padding(padding);
        Conv3d conv2 = new Conv3d(conv2Options);
        block.add(conv2);

        // Batch normalization
        BatchNorm3d bn2 = new BatchNorm3d(outChannels);
        block.add(bn2);

        // Leaky ReLU activation
        LeakyReLUOptions relu2Options = new LeakyReLUOptions();
        relu2Options.negative_slope(0.2);
        LeakyReLU relu2 = new LeakyReLU(relu2Options);
        block.add(relu2);

        return block;
    }

    /**
     * Forward pass through the encoder.
     *
     * @param x Input tensor [B, C, D, H, W]
     * @return EncoderOutput containing μ and log σ² tensors
     */
    public EncoderOutput forward(Tensor x) {
        // Validate input
        if (x == null) {
            throw new IllegalArgumentException("Input tensor is null");
        }

        long[] shape = x.sizes();
        if (shape.length != 5) {
            throw new IllegalArgumentException(
                String.format("Expected 5D tensor [B,C,D,H,W], got %dD: %s",
                            shape.length, java.util.Arrays.toString(shape)));
        }

        if (shape[1] != inputChannels) {
            throw new IllegalArgumentException(
                String.format("Expected %d input channels, got %d",
                            inputChannels, shape[1]));
        }

        logger.debug("Encoder forward: input shape = {}", java.util.Arrays.toString(shape));

        // Convolutional feature extraction
        Tensor features = convBlocks.forward(x);

        logger.debug("After conv blocks: shape = {}",
                    java.util.Arrays.toString(features.sizes()));

        // Flatten
        long batchSize = features.size(0);
        Tensor flattened = features.view(new long[]{batchSize, -1});

        logger.debug("After flatten: shape = {}",
                    java.util.Arrays.toString(flattened.sizes()));

        // Compute latent distribution parameters
        Tensor mu = fcMu.forward(flattened);
        Tensor logVar = fcLogVar.forward(flattened);

        logger.debug("Encoder output: mu shape = {}, logVar shape = {}",
                    java.util.Arrays.toString(mu.sizes()),
                    java.util.Arrays.toString(logVar.sizes()));

        return new EncoderOutput(mu, logVar);
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
     * Gets the input channels.
     *
     * @return Number of input channels
     */
    public int getInputChannels() {
        return inputChannels;
    }

    /**
     * Gets the channel progression.
     *
     * @return Array of channel sizes
     */
    public int[] getChannels() {
        return channels.clone();
    }

    /**
     * Container for encoder output (latent distribution parameters).
     */
    public static class EncoderOutput {
        /** Mean of latent distribution [B, latentDim] */
        public final Tensor mu;

        /** Log variance of latent distribution [B, latentDim] */
        public final Tensor logVar;

        /**
         * Creates encoder output.
         *
         * @param mu Mean tensor
         * @param logVar Log variance tensor
         */
        public EncoderOutput(Tensor mu, Tensor logVar) {
            this.mu = mu;
            this.logVar = logVar;
        }

        /**
         * Computes standard deviation from log variance.
         *
         * @return Standard deviation tensor
         */
        public Tensor getStd() {
            // σ = exp(0.5 * log(σ²))
            return logVar.mul(0.5).exp();
        }

        /**
         * Computes variance from log variance.
         *
         * @return Variance tensor
         */
        public Tensor getVar() {
            return logVar.exp();
        }
    }

    @Override
    public String toString() {
        return String.format("VAEEncoder3D(input_channels=%d, latent_dim=%d, channels=%s)",
                           inputChannels, latentDim, java.util.Arrays.toString(channels));
    }
}
