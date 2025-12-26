package vtea.deeplearning.loss;

import org.bytedeco.pytorch.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import vtea.deeplearning.models.VAEConfig;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Reconstruction loss for Variational Autoencoder.
 *
 * <p>Measures how well the decoder reconstructs the original input
 * from the latent representation.</p>
 *
 * <p>Supported loss types:</p>
 * <ul>
 *   <li><b>MSE (Mean Squared Error):</b> For continuous-valued data</li>
 *   <li><b>BCE (Binary Cross-Entropy):</b> For normalized [0,1] data</li>
 *   <li><b>L1 (Mean Absolute Error):</b> More robust to outliers</li>
 * </ul>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class ReconstructionLoss {

    private static final Logger logger = LoggerFactory.getLogger(ReconstructionLoss.class);

    private final VAEConfig.ReconstructionType lossType;

    /**
     * Creates a reconstruction loss with the specified type.
     *
     * @param lossType Type of reconstruction loss (MSE, BCE, L1)
     */
    public ReconstructionLoss(VAEConfig.ReconstructionType lossType) {
        if (lossType == null) {
            throw new IllegalArgumentException("Loss type cannot be null");
        }

        this.lossType = lossType;
        logger.info("ReconstructionLoss initialized with type: {}", lossType);
    }

    /**
     * Computes reconstruction loss between reconstruction and target.
     *
     * @param reconstruction Reconstructed tensor [B, C, D, H, W]
     * @param target Original input tensor [B, C, D, H, W]
     * @return Scalar reconstruction loss
     */
    public Tensor compute(Tensor reconstruction, Tensor target) {
        if (reconstruction == null || target == null) {
            throw new IllegalArgumentException("reconstruction and target cannot be null");
        }

        // Validate shapes match
        long[] reconShape = reconstruction.sizes();
        long[] targetShape = target.sizes();

        if (reconShape.length != targetShape.length) {
            throw new IllegalArgumentException(
                String.format("Dimension mismatch: reconstruction=%dD, target=%dD",
                            reconShape.length, targetShape.length));
        }

        for (int i = 0; i < reconShape.length; i++) {
            if (reconShape[i] != targetShape[i]) {
                throw new IllegalArgumentException(
                    String.format("Shape mismatch at dim %d: reconstruction=%s, target=%s",
                                i, java.util.Arrays.toString(reconShape),
                                java.util.Arrays.toString(targetShape)));
            }
        }

        Tensor loss;

        switch (lossType) {
            case MSE:
                loss = computeMSE(reconstruction, target);
                break;

            case BCE:
                loss = computeBCE(reconstruction, target);
                break;

            case L1:
                loss = computeL1(reconstruction, target);
                break;

            default:
                throw new IllegalStateException("Unknown loss type: " + lossType);
        }

        logger.debug("{} reconstruction loss: {}", lossType, loss.item().doubleValue());

        return loss;
    }

    /**
     * Computes Mean Squared Error (MSE) loss.
     *
     * <p>MSE = mean((reconstruction - target)²)</p>
     *
     * <p>Good for continuous-valued data. Sensitive to outliers.</p>
     *
     * @param reconstruction Reconstructed tensor
     * @param target Original tensor
     * @return MSE loss
     */
    private Tensor computeMSE(Tensor reconstruction, Tensor target) {
        // PyTorch mse_loss with reduction='mean'
        return mse_loss(reconstruction, target, Reduction.Mean);
    }

    /**
     * Computes Binary Cross-Entropy (BCE) loss.
     *
     * <p>BCE = -mean(target × log(reconstruction) + (1-target) × log(1-reconstruction))</p>
     *
     * <p>Good for binary or normalized [0,1] data. Handles saturation better than MSE.</p>
     *
     * @param reconstruction Reconstructed tensor (should be in [0,1] via sigmoid)
     * @param target Original tensor (should be in [0,1])
     * @return BCE loss
     */
    private Tensor computeBCE(Tensor reconstruction, Tensor target) {
        // PyTorch binary_cross_entropy with reduction='mean'
        // Note: reconstruction should already have sigmoid applied
        return binary_cross_entropy(reconstruction, target, (Tensor) null, Reduction.Mean);
    }

    /**
     * Computes L1 (Mean Absolute Error) loss.
     *
     * <p>L1 = mean(|reconstruction - target|)</p>
     *
     * <p>More robust to outliers than MSE. Produces sharper reconstructions.</p>
     *
     * @param reconstruction Reconstructed tensor
     * @param target Original tensor
     * @return L1 loss
     */
    private Tensor computeL1(Tensor reconstruction, Tensor target) {
        // PyTorch l1_loss with reduction='mean'
        return l1_loss(reconstruction, target, Reduction.Mean);
    }

    /**
     * Computes per-sample reconstruction loss.
     *
     * <p>Returns loss for each sample in the batch,
     * useful for identifying poorly reconstructed samples.</p>
     *
     * @param reconstruction Reconstructed tensor [B, C, D, H, W]
     * @param target Original tensor [B, C, D, H, W]
     * @return Per-sample loss [B]
     */
    public Tensor computePerSample(Tensor reconstruction, Tensor target) {
        Tensor diff;

        switch (lossType) {
            case MSE:
                diff = reconstruction.sub(target).pow(2.0);
                break;

            case BCE:
                // BCE per element
                Tensor term1 = target.mul(reconstruction.log());
                Tensor term2 = target.mul(-1.0).add(1.0)
                                    .mul(reconstruction.mul(-1.0).add(1.0).log());
                diff = term1.add(term2).mul(-1.0);
                break;

            case L1:
                diff = reconstruction.sub(target).abs();
                break;

            default:
                throw new IllegalStateException("Unknown loss type: " + lossType);
        }

        // Mean over all dimensions except batch
        // For 5D tensor [B, C, D, H, W], mean over dimensions 1,2,3,4
        long[] shape = diff.sizes();
        long[] reduceDims = new long[shape.length - 1];
        for (int i = 0; i < reduceDims.length; i++) {
            reduceDims[i] = i + 1;
        }

        Tensor perSampleLoss = diff.mean(reduceDims);

        return perSampleLoss;
    }

    /**
     * Computes SSIM (Structural Similarity Index) as additional metric.
     *
     * <p>SSIM measures structural similarity between images,
     * often correlates better with human perception than MSE.</p>
     *
     * <p>Note: This is a placeholder. Full SSIM implementation
     * requires sliding window operations.</p>
     *
     * @param reconstruction Reconstructed tensor
     * @param target Original tensor
     * @return SSIM value (higher is better, range [0,1])
     */
    public double computeSSIM(Tensor reconstruction, Tensor target) {
        // Placeholder for SSIM implementation
        // Full implementation would require:
        // 1. Sliding window convolutions
        // 2. Local mean, variance, covariance computations
        // 3. SSIM formula application per window
        // 4. Averaging over all windows

        logger.warn("SSIM computation not yet fully implemented");
        return 0.0;
    }

    /**
     * Gets the reconstruction loss type.
     *
     * @return Loss type
     */
    public VAEConfig.ReconstructionType getLossType() {
        return lossType;
    }

    @Override
    public String toString() {
        return String.format("ReconstructionLoss(type=%s)", lossType);
    }
}
