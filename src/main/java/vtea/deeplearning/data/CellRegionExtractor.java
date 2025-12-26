package vtea.deeplearning.data;

import ij.ImageStack;
import ij.process.FloatProcessor;
import vteaobjects.MicroObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Extracts 3D cubic regions centered on cell centroids from volumetric image data.
 *
 * <p>This class is responsible for extracting fixed-size 3D regions around
 * segmented cells (MicroObjects) from the full image volume. These regions
 * are used as input to deep learning models.</p>
 *
 * <p>Features:</p>
 * <ul>
 *   <li>Configurable region size (32³, 64³, 128³)</li>
 *   <li>Multiple padding strategies for boundary cells</li>
 *   <li>Multi-channel support</li>
 *   <li>Automatic centering on cell centroid</li>
 * </ul>
 *
 * @author VTEA Development Team
 * @version 1.0
 */
public class CellRegionExtractor {

    private static final Logger logger = LoggerFactory.getLogger(CellRegionExtractor.class);

    private final int regionSize;
    private final PaddingType paddingType;

    /**
     * Padding strategies for regions that extend beyond image boundaries.
     */
    public enum PaddingType {
        /** Pad with zeros */
        ZERO,

        /** Mirror reflect at boundaries */
        MIRROR,

        /** Replicate edge values */
        REPLICATE,

        /** Crop to available region (may result in smaller output) */
        CROP
    }

    /**
     * Creates a CellRegionExtractor with specified region size and padding.
     *
     * @param regionSize The size of the cubic region to extract (e.g., 64 for 64³)
     * @param paddingType The padding strategy for boundary regions
     */
    public CellRegionExtractor(int regionSize, PaddingType paddingType) {
        if (regionSize <= 0 || !isPowerOfTwo(regionSize)) {
            throw new IllegalArgumentException(
                "Region size must be a positive power of 2, got: " + regionSize);
        }

        this.regionSize = regionSize;
        this.paddingType = paddingType;

        logger.info("CellRegionExtractor initialized: regionSize={}, padding={}",
                   regionSize, paddingType);
    }

    /**
     * Creates a CellRegionExtractor with default settings (64³, MIRROR padding).
     */
    public CellRegionExtractor() {
        this(64, PaddingType.MIRROR);
    }

    /**
     * Extracts a 3D region around a cell from a single-channel image.
     *
     * @param cell The MicroObject representing the segmented cell
     * @param imageStack The source image volume
     * @return ImageStack containing the extracted region (regionSize³)
     */
    public ImageStack extractRegion(MicroObject cell, ImageStack imageStack) {
        if (cell == null) {
            throw new IllegalArgumentException("MicroObject is null");
        }
        if (imageStack == null) {
            throw new IllegalArgumentException("ImageStack is null");
        }

        // Get centroid coordinates
        double[] centroid = cell.getCentroidGlobal();
        int cx = (int) Math.round(centroid[0]);
        int cy = (int) Math.round(centroid[1]);
        int cz = (int) Math.round(centroid[2]);

        logger.debug("Extracting region for cell at ({}, {}, {})", cx, cy, cz);

        // Calculate region bounds
        int halfSize = regionSize / 2;
        int x0 = cx - halfSize;
        int y0 = cy - halfSize;
        int z0 = cz - halfSize;
        int x1 = x0 + regionSize;
        int y1 = y0 + regionSize;
        int z1 = z0 + regionSize;

        // Extract region
        return extractRegion(imageStack, x0, y0, z0, x1, y1, z1);
    }

    /**
     * Extracts a 3D region from multi-channel images.
     *
     * @param cell The MicroObject representing the segmented cell
     * @param imageStacks Array of ImageStacks, one per channel
     * @return Array of ImageStacks containing the extracted regions
     */
    public ImageStack[] extractRegion(MicroObject cell, ImageStack[] imageStacks) {
        if (imageStacks == null || imageStacks.length == 0) {
            throw new IllegalArgumentException("ImageStacks array is null or empty");
        }

        ImageStack[] regions = new ImageStack[imageStacks.length];

        for (int c = 0; c < imageStacks.length; c++) {
            regions[c] = extractRegion(cell, imageStacks[c]);
        }

        return regions;
    }

    /**
     * Extracts a 3D region with specified bounds from an ImageStack.
     *
     * @param source Source ImageStack
     * @param x0 Start x coordinate (inclusive)
     * @param y0 Start y coordinate (inclusive)
     * @param z0 Start z coordinate (inclusive, 0-indexed)
     * @param x1 End x coordinate (exclusive)
     * @param y1 End y coordinate (exclusive)
     * @param z1 End z coordinate (exclusive, 0-indexed)
     * @return Extracted region as ImageStack
     */
    private ImageStack extractRegion(ImageStack source,
                                    int x0, int y0, int z0,
                                    int x1, int y1, int z1) {
        int srcWidth = source.getWidth();
        int srcHeight = source.getHeight();
        int srcDepth = source.getSize();

        int dstWidth = x1 - x0;
        int dstHeight = y1 - y0;
        int dstDepth = z1 - z0;

        ImageStack destination = new ImageStack(dstWidth, dstHeight);

        // Extract each slice
        for (int z = 0; z < dstDepth; z++) {
            FloatProcessor fp = new FloatProcessor(dstWidth, dstHeight);

            for (int y = 0; y < dstHeight; y++) {
                for (int x = 0; x < dstWidth; x++) {
                    int srcX = x0 + x;
                    int srcY = y0 + y;
                    int srcZ = z0 + z;

                    float value = getPixelValue(source, srcX, srcY, srcZ,
                                               srcWidth, srcHeight, srcDepth);
                    fp.setf(x, y, value);
                }
            }

            destination.addSlice("z=" + (z + 1), fp);
        }

        return destination;
    }

    /**
     * Gets pixel value with boundary handling according to padding type.
     */
    private float getPixelValue(ImageStack stack, int x, int y, int z,
                               int width, int height, int depth) {
        // Check if coordinates are within bounds
        boolean inBounds = (x >= 0 && x < width &&
                           y >= 0 && y < height &&
                           z >= 0 && z < depth);

        if (inBounds) {
            // Direct access (z is 0-indexed for our purposes, but ImageStack is 1-indexed)
            return stack.getProcessor(z + 1).getf(x, y);
        }

        // Handle out-of-bounds according to padding type
        switch (paddingType) {
            case ZERO:
                return 0.0f;

            case MIRROR:
                return getMirrorPixel(stack, x, y, z, width, height, depth);

            case REPLICATE:
                return getReplicatePixel(stack, x, y, z, width, height, depth);

            case CROP:
                // For CROP, we should have already handled this at extraction level
                return 0.0f;

            default:
                throw new IllegalStateException("Unknown padding type: " + paddingType);
        }
    }

    /**
     * Gets pixel value using mirror reflection at boundaries.
     */
    private float getMirrorPixel(ImageStack stack, int x, int y, int z,
                                int width, int height, int depth) {
        // Mirror reflect coordinates
        x = mirrorCoordinate(x, width);
        y = mirrorCoordinate(y, height);
        z = mirrorCoordinate(z, depth);

        return stack.getProcessor(z + 1).getf(x, y);
    }

    /**
     * Gets pixel value by replicating edge values.
     */
    private float getReplicatePixel(ImageStack stack, int x, int y, int z,
                                   int width, int height, int depth) {
        // Clamp coordinates to valid range
        x = Math.max(0, Math.min(x, width - 1));
        y = Math.max(0, Math.min(y, height - 1));
        z = Math.max(0, Math.min(z, depth - 1));

        return stack.getProcessor(z + 1).getf(x, y);
    }

    /**
     * Mirrors a coordinate to fall within valid range.
     */
    private int mirrorCoordinate(int coord, int size) {
        if (coord < 0) {
            // Mirror on left/top side
            coord = -coord - 1;
        } else if (coord >= size) {
            // Mirror on right/bottom side
            coord = 2 * size - coord - 1;
        }

        // Ensure still in bounds (for multiple reflections)
        coord = Math.max(0, Math.min(coord, size - 1));

        return coord;
    }

    /**
     * Checks if a number is a power of two.
     */
    private boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    /**
     * Gets the configured region size.
     *
     * @return The cubic region size
     */
    public int getRegionSize() {
        return regionSize;
    }

    /**
     * Gets the configured padding type.
     *
     * @return The padding strategy
     */
    public PaddingType getPaddingType() {
        return paddingType;
    }

    /**
     * Estimates if a cell's region will require padding.
     *
     * @param cell The cell to check
     * @param imageStack The source image
     * @return true if padding will be needed
     */
    public boolean requiresPadding(MicroObject cell, ImageStack imageStack) {
        double[] centroid = cell.getCentroidGlobal();
        int cx = (int) Math.round(centroid[0]);
        int cy = (int) Math.round(centroid[1]);
        int cz = (int) Math.round(centroid[2]);

        int halfSize = regionSize / 2;

        int x0 = cx - halfSize;
        int y0 = cy - halfSize;
        int z0 = cz - halfSize;
        int x1 = x0 + regionSize;
        int y1 = y0 + regionSize;
        int z1 = z0 + regionSize;

        boolean needsPadding = (x0 < 0 || y0 < 0 || z0 < 0 ||
                                x1 > imageStack.getWidth() ||
                                y1 > imageStack.getHeight() ||
                                z1 > imageStack.getSize());

        if (needsPadding) {
            logger.debug("Cell at ({}, {}, {}) requires padding", cx, cy, cz);
        }

        return needsPadding;
    }

    /**
     * Calculates the bounding box for a cell's extraction region.
     *
     * @param cell The cell
     * @return int array [x0, y0, z0, x1, y1, z1]
     */
    public int[] getRegionBounds(MicroObject cell) {
        double[] centroid = cell.getCentroidGlobal();
        int cx = (int) Math.round(centroid[0]);
        int cy = (int) Math.round(centroid[1]);
        int cz = (int) Math.round(centroid[2]);

        int halfSize = regionSize / 2;

        return new int[]{
            cx - halfSize,  // x0
            cy - halfSize,  // y0
            cz - halfSize,  // z0
            cx + halfSize,  // x1
            cy + halfSize,  // y1
            cz + halfSize   // z1
        };
    }
}
