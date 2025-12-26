package vtea.deeplearning.models;

import org.bytedeco.pytorch.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * 3D Transpose Convolutional Decoder for Variational Autoencoder.
 *
 * <p>This decoder network takes latent vectors and progressively upsamples them
 * through transpose convolutional blocks to reconstruct the original 3D volumes.</p>
 *
 * <p>Architecture (mirrors the encoder):</p>
 * <pre>
 * Latent vector z [B, latentDim]
 *   ↓
 * FC → Reshape to [B, channels[N-1], D', H', W']
 *   ↓
 * TransposeConv3D Block N: channels[N-1] → channels[N-2]
 *   ↓
 * ...
 *   ↓
 * TransposeConv3D Block 1: channels[0] → outputChannels
 *   ↓
 * Sigmoid (for [0,1] output)
 *   ↓
 * Reconstruction [B, C, D, H, W]
 * </pre>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class VAEDecoder3D extends Module {

    private static final Logger logger = LoggerFactory.getLogger(VAEDecoder3D.class);

    private final int latentDim;
    private final int outputChannels;
    private final int[] channels;
    private final int outputSize;

    private Linear fcProject;
    private Sequential deconvBlocks;
    private Sigmoid finalActivation;

    private long[] reshapeSize; // Shape after FC projection

    /**
     * Creates a 3D decoder for VAE.
     *
     * @param latentDim Dimensionality of latent space
     * @param outputChannels Number of output channels (should match input channels)
     * @param channels Array specifying channel progression (same as encoder)
     * @param outputSize Target output spatial size (e.g., 64 for 64³)
     */
    public VAEDecoder3D(int latentDim, int outputChannels, int[] channels, int outputSize) {
        super();

        if (latentDim <= 0) {
            throw new IllegalArgumentException("Latent dimension must be positive");
        }
        if (outputChannels <= 0) {
            throw new IllegalArgumentException("Output channels must be positive");
        }
        if (channels == null || channels.length == 0) {
            throw new IllegalArgumentException("Channels array must not be empty");
        }
        if (outputSize <= 0) {
            throw new IllegalArgumentException("Output size must be positive");
        }

        this.latentDim = latentDim;
        this.outputChannels = outputChannels;
        this.channels = channels;
        this.outputSize = outputSize;

        logger.info("Creating VAE Decoder: latent_dim={}, output_channels={}, " +
                   "channels={}, output_size={}",
                   latentDim, outputChannels,
                   java.util.Arrays.toString(channels), outputSize);

        buildNetwork();
    }

    /**
     * Builds the decoder network architecture.
     */
    private void buildNetwork() {
        // Calculate initial spatial size after FC projection
        // If we have N blocks with stride 2, and final size is S, initial size ≈ S / 2^N
        int numStrides = channels.length - 1; // Number of upsampling blocks
        int initialSpatialSize = outputSize / (int) Math.pow(2, numStrides);

        if (initialSpatialSize < 1) {
            initialSpatialSize = 4; // Minimum initial size
            logger.warn("Calculated initial spatial size < 1, using minimum: {}",
                       initialSpatialSize);
        }

        int projectionChannels = channels[channels.length - 1]; // Highest channel count
        int projectionSize = projectionChannels *
                           initialSpatialSize * initialSpatialSize * initialSpatialSize;

        reshapeSize = new long[]{
            projectionChannels,
            initialSpatialSize,
            initialSpatialSize,
            initialSpatialSize
        };

        logger.info("Decoder projection: latent {} → flattened {} → reshape to {}",
                   latentDim, projectionSize, java.util.Arrays.toString(reshapeSize));

        // FC projection from latent to feature space
        fcProject = new Linear(latentDim, projectionSize);
        register_module("fc_project", fcProject);

        // Build transpose convolutional blocks
        deconvBlocks = buildDeconvolutionalBlocks();
        register_module("deconv_blocks", deconvBlocks);

        // Final sigmoid activation for [0, 1] output range
        finalActivation = new Sigmoid();
        register_module("final_activation", finalActivation);

        logger.info("Decoder built successfully");
    }

    /**
     * Builds sequential transpose convolutional blocks.
     *
     * @return Sequential module containing all deconv blocks
     */
    private Sequential buildDeconvolutionalBlocks() {
        Sequential blocks = new Sequential();

        // Reverse the channel progression for decoding
        int[] reversedChannels = reverseArray(channels);

        for (int i = 0; i < reversedChannels.length; i++) {
            int inChannels = reversedChannels[i];
            int outChannels = (i == reversedChannels.length - 1) ?
                             outputChannels : reversedChannels[i + 1];

            boolean upsample = (i < reversedChannels.length - 1);

            Sequential block = createDeconvBlock(inChannels, outChannels, upsample);
            blocks.add(block);

            logger.debug("Added deconv block {}: {} → {} channels, upsample={}",
                        i, inChannels, outChannels, upsample);
        }

        return blocks;
    }

    /**
     * Creates a single transpose convolutional block.
     *
     * <p>Each block contains:</p>
     * <ul>
     *   <li>ConvTranspose3d (with optional stride=2 for upsampling)</li>
     *   <li>BatchNorm3d (except for final block)</li>
     *   <li>LeakyReLU (except for final block)</li>
     * </ul>
     *
     * @param inChannels Input channels
     * @param outChannels Output channels
     * @param upsample Whether to upsample (stride=2)
     * @return Sequential block
     */
    private Sequential createDeconvBlock(int inChannels, int outChannels, boolean upsample) {
        Sequential block = new Sequential();

        long[] kernelSize = {3, 3, 3};
        long[] padding = {1, 1, 1};
        long[] outputPadding = upsample ? new long[]{1, 1, 1} : new long[]{0, 0, 0};
        int stride = upsample ? 2 : 1;

        // Transpose convolution
        ConvTranspose3dOptions deconvOptions =
            new ConvTranspose3dOptions(inChannels, outChannels, kernelSize);
        deconvOptions.stride(new long[]{stride, stride, stride});
        deconvOptions.padding(padding);
        deconvOptions.output_padding(outputPadding);

        ConvTranspose3d deconv = new ConvTranspose3d(deconvOptions);
        block.add(deconv);

        // Only add batch norm and activation if not final layer
        if (upsample) {
            // Batch normalization
            BatchNorm3d bn = new BatchNorm3d(outChannels);
            block.add(bn);

            // Leaky ReLU activation
            LeakyReLUOptions reluOptions = new LeakyReLUOptions();
            reluOptions.negative_slope(0.2);
            LeakyReLU relu = new LeakyReLU(reluOptions);
            block.add(relu);
        }

        return block;
    }

    /**
     * Forward pass through the decoder.
     *
     * @param z Latent vector [B, latentDim]
     * @return Reconstructed volume [B, outputChannels, outputSize, outputSize, outputSize]
     */
    public Tensor forward(Tensor z) {
        // Validate input
        if (z == null) {
            throw new IllegalArgumentException("Input latent tensor is null");
        }

        long[] shape = z.sizes();
        if (shape.length != 2) {
            throw new IllegalArgumentException(
                String.format("Expected 2D tensor [B, latentDim], got %dD: %s",
                            shape.length, java.util.Arrays.toString(shape)));
        }

        if (shape[1] != latentDim) {
            throw new IllegalArgumentException(
                String.format("Expected latent dim %d, got %d",
                            latentDim, shape[1]));
        }

        long batchSize = shape[0];

        logger.debug("Decoder forward: input shape = {}", java.util.Arrays.toString(shape));

        // Project to feature space
        Tensor projected = fcProject.forward(z);

        logger.debug("After FC projection: shape = {}",
                    java.util.Arrays.toString(projected.sizes()));

        // Reshape to 3D
        long[] reshapeTarget = new long[]{
            batchSize,
            reshapeSize[0],  // channels
            reshapeSize[1],  // depth
            reshapeSize[2],  // height
            reshapeSize[3]   // width
        };

        Tensor reshaped = projected.view(reshapeTarget);

        logger.debug("After reshape: shape = {}",
                    java.util.Arrays.toString(reshaped.sizes()));

        // Transpose convolutional upsampling
        Tensor upsampled = deconvBlocks.forward(reshaped);

        logger.debug("After deconv blocks: shape = {}",
                    java.util.Arrays.toString(upsampled.sizes()));

        // Apply sigmoid activation for [0, 1] range
        Tensor reconstruction = finalActivation.forward(upsampled);

        logger.debug("Final reconstruction: shape = {}",
                    java.util.Arrays.toString(reconstruction.sizes()));

        return reconstruction;
    }

    /**
     * Reverses an array.
     *
     * @param arr Input array
     * @return Reversed array
     */
    private int[] reverseArray(int[] arr) {
        int[] reversed = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            reversed[i] = arr[arr.length - 1 - i];
        }
        return reversed;
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
     * Gets the output channels.
     *
     * @return Number of output channels
     */
    public int getOutputChannels() {
        return outputChannels;
    }

    /**
     * Gets the output spatial size.
     *
     * @return Output size (for cubic volumes)
     */
    public int getOutputSize() {
        return outputSize;
    }

    /**
     * Gets the channel progression.
     *
     * @return Array of channel sizes
     */
    public int[] getChannels() {
        return channels.clone();
    }

    @Override
    public String toString() {
        return String.format("VAEDecoder3D(latent_dim=%d, output_channels=%d, " +
                           "channels=%s, output_size=%d)",
                           latentDim, outputChannels,
                           java.util.Arrays.toString(channels), outputSize);
    }
}
