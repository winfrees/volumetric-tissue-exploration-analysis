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
package vtea.deeplearning.data;

import ij.ImageStack;
import ij.process.ImageProcessor;
import vteaobjects.MicroObject;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for extracting 3D cubic regions around segmented cells (MicroObjects).
 * Handles boundary cases with various padding strategies.
 *
 * Supports two APIs:
 * 1. Static methods for multi-channel extraction (classification workflows)
 * 2. Instance methods for single-channel extraction (VAE workflows)
 *
 * @author VTEA Deep Learning Team
 */
public class CellRegionExtractor {

    // Instance fields for VAE-style API
    private final int regionSize;
    private final PaddingType paddingType;

    /**
     * Padding strategies for handling boundary cases
     */
    public enum PaddingType {
        ZERO,       // Pad with zeros
        MIRROR,     // Mirror the edge pixels
        REPLICATE   // Replicate the edge pixels
    }

    /**
     * Extract a 3D cubic region around a MicroObject centroid from multi-channel ImageStacks.
     *
     * @param object       MicroObject to extract region around
     * @param imageStacks  Array of ImageStacks (one per channel)
     * @param regionSize   Size of the region [depth, height, width]
     * @param channels     Channel indices to extract (null = all channels)
     * @param paddingType  How to handle boundaries
     * @return Array of ImageStacks (one per channel) containing the extracted region
     */
    public static ImageStack[] extractRegion(MicroObject object, ImageStack[] imageStacks,
                                              int[] regionSize, int[] channels,
                                              PaddingType paddingType) {
        if (object == null) {
            throw new IllegalArgumentException("MicroObject cannot be null");
        }
        if (imageStacks == null || imageStacks.length == 0) {
            throw new IllegalArgumentException("ImageStacks array cannot be null or empty");
        }
        if (regionSize == null || regionSize.length != 3) {
            throw new IllegalArgumentException("Region size must be [depth, height, width]");
        }

        // Determine which channels to extract
        int[] channelIndices = channels;
        if (channelIndices == null) {
            channelIndices = new int[imageStacks.length];
            for (int i = 0; i < channelIndices.length; i++) {
                channelIndices[i] = i;
            }
        }

        // Get object centroid
        float centroidX = object.getCentroidX();
        float centroidY = object.getCentroidY();
        float centroidZ = object.getCentroidZ();

        // Calculate bounding box for extraction
        int halfDepth = regionSize[0] / 2;
        int halfHeight = regionSize[1] / 2;
        int halfWidth = regionSize[2] / 2;

        int startZ = Math.round(centroidZ) - halfDepth;
        int startY = Math.round(centroidY) - halfHeight;
        int startX = Math.round(centroidX) - halfWidth;

        int endZ = startZ + regionSize[0];
        int endY = startY + regionSize[1];
        int endX = startX + regionSize[2];

        // Get image dimensions
        int imgWidth = imageStacks[0].getWidth();
        int imgHeight = imageStacks[0].getHeight();
        int imgDepth = imageStacks[0].getSize();

        // Extract region for each channel
        ImageStack[] extractedRegions = new ImageStack[channelIndices.length];

        for (int c = 0; c < channelIndices.length; c++) {
            ImageStack sourceStack = imageStacks[channelIndices[c]];
            ImageStack regionStack = new ImageStack(regionSize[2], regionSize[1], regionSize[0]);

            for (int z = 0; z < regionSize[0]; z++) {
                int sourceZ = startZ + z;
                ImageProcessor sliceProcessor = new ij.process.FloatProcessor(regionSize[2], regionSize[1]);

                for (int y = 0; y < regionSize[1]; y++) {
                    int sourceY = startY + y;

                    for (int x = 0; x < regionSize[2]; x++) {
                        int sourceX = startX + x;

                        float value;
                        if (sourceX >= 0 && sourceX < imgWidth &&
                            sourceY >= 0 && sourceY < imgHeight &&
                            sourceZ >= 0 && sourceZ < imgDepth) {
                            // Inside image bounds - get actual pixel value
                            ImageProcessor sourceProc = sourceStack.getProcessor(sourceZ + 1);
                            value = sourceProc.getPixelValue(sourceX, sourceY);
                        } else {
                            // Outside image bounds - apply padding
                            value = getPaddedValue(sourceStack, sourceX, sourceY, sourceZ,
                                                    imgWidth, imgHeight, imgDepth, paddingType);
                        }

                        sliceProcessor.setf(x, y, value);
                    }
                }

                regionStack.setProcessor(sliceProcessor, z + 1);
            }

            extractedRegions[c] = regionStack;
        }

        return extractedRegions;
    }

    /**
     * Convenience method with default padding type (ZERO)
     */
    public static ImageStack[] extractRegion(MicroObject object, ImageStack[] imageStacks,
                                              int[] regionSize, int[] channels) {
        return extractRegion(object, imageStacks, regionSize, channels, PaddingType.ZERO);
    }

    /**
     * Extract regions for multiple objects in batch
     *
     * @param objects      List of MicroObjects
     * @param imageStacks  Multi-channel image stacks
     * @param regionSize   Size of regions to extract
     * @param channels     Channels to include
     * @param paddingType  Padding strategy
     * @return List of extracted region arrays
     */
    public static List<ImageStack[]> extractBatch(List<MicroObject> objects,
                                                    ImageStack[] imageStacks,
                                                    int[] regionSize,
                                                    int[] channels,
                                                    PaddingType paddingType) {
        List<ImageStack[]> batch = new ArrayList<>();

        for (MicroObject object : objects) {
            ImageStack[] region = extractRegion(object, imageStacks, regionSize, channels, paddingType);
            batch.add(region);
        }

        return batch;
    }

    /**
     * Get padded value for coordinates outside image bounds
     */
    private static float getPaddedValue(ImageStack stack, int x, int y, int z,
                                         int width, int height, int depth,
                                         PaddingType paddingType) {
        switch (paddingType) {
            case ZERO:
                return 0.0f;

            case MIRROR:
                // Mirror coordinates at boundaries
                int mirrorX = mirrorCoordinate(x, width);
                int mirrorY = mirrorCoordinate(y, height);
                int mirrorZ = mirrorCoordinate(z, depth);
                ImageProcessor proc = stack.getProcessor(mirrorZ + 1);
                return proc.getPixelValue(mirrorX, mirrorY);

            case REPLICATE:
                // Clamp coordinates to valid range
                int clampX = Math.max(0, Math.min(x, width - 1));
                int clampY = Math.max(0, Math.min(y, height - 1));
                int clampZ = Math.max(0, Math.min(z, depth - 1));
                ImageProcessor clampProc = stack.getProcessor(clampZ + 1);
                return clampProc.getPixelValue(clampX, clampY);

            default:
                return 0.0f;
        }
    }

    /**
     * Mirror coordinate at boundaries
     * Example: -1 -> 1, -2 -> 2, width -> width-2, width+1 -> width-1
     */
    private static int mirrorCoordinate(int coord, int size) {
        if (coord < 0) {
            return Math.abs(coord);
        } else if (coord >= size) {
            return 2 * size - coord - 2;
        } else {
            return coord;
        }
    }

    /**
     * Check if a region can be extracted without padding
     *
     * @param object      MicroObject to check
     * @param imageStacks Image stacks
     * @param regionSize  Desired region size
     * @return true if region fits entirely within image bounds
     */
    public static boolean canExtractWithoutPadding(MicroObject object, ImageStack[] imageStacks,
                                                     int[] regionSize) {
        if (imageStacks == null || imageStacks.length == 0) {
            return false;
        }

        float centroidX = object.getCentroidX();
        float centroidY = object.getCentroidY();
        float centroidZ = object.getCentroidZ();

        int halfDepth = regionSize[0] / 2;
        int halfHeight = regionSize[1] / 2;
        int halfWidth = regionSize[2] / 2;

        int startZ = Math.round(centroidZ) - halfDepth;
        int startY = Math.round(centroidY) - halfHeight;
        int startX = Math.round(centroidX) - halfWidth;

        int endZ = startZ + regionSize[0];
        int endY = startY + regionSize[1];
        int endX = startX + regionSize[2];

        int imgWidth = imageStacks[0].getWidth();
        int imgHeight = imageStacks[0].getHeight();
        int imgDepth = imageStacks[0].getSize();

        return startX >= 0 && endX <= imgWidth &&
               startY >= 0 && endY <= imgHeight &&
               startZ >= 0 && endZ <= imgDepth;
    }

    /**
     * Get count of objects that require padding
     */
    public static int countObjectsRequiringPadding(List<MicroObject> objects,
                                                     ImageStack[] imageStacks,
                                                     int[] regionSize) {
        int count = 0;
        for (MicroObject object : objects) {
            if (!canExtractWithoutPadding(object, imageStacks, regionSize)) {
                count++;
            }
        }
        return count;
    }

    /**
     * Filter objects to only include those that can be extracted without padding
     */
    public static List<MicroObject> filterObjectsWithinBounds(List<MicroObject> objects,
                                                                ImageStack[] imageStacks,
                                                                int[] regionSize) {
        List<MicroObject> filtered = new ArrayList<>();
        for (MicroObject object : objects) {
            if (canExtractWithoutPadding(object, imageStacks, regionSize)) {
                filtered.add(object);
            }
        }
        return filtered;
    }

    /**
     * Calculate optimal region size for a given object and image
     * Returns the largest cubic region that can be extracted without padding
     *
     * @param object      MicroObject
     * @param imageStacks Image stacks
     * @param maxSize     Maximum desired size
     * @return Optimal region size [depth, height, width]
     */
    public static int[] calculateOptimalRegionSize(MicroObject object, ImageStack[] imageStacks,
                                                     int maxSize) {
        if (imageStacks == null || imageStacks.length == 0) {
            return new int[]{maxSize, maxSize, maxSize};
        }

        float centroidX = object.getCentroidX();
        float centroidY = object.getCentroidY();
        float centroidZ = object.getCentroidZ();

        int imgWidth = imageStacks[0].getWidth();
        int imgHeight = imageStacks[0].getHeight();
        int imgDepth = imageStacks[0].getSize();

        // Calculate maximum distance to boundaries in each dimension
        int maxDistX = Math.min((int) centroidX, imgWidth - (int) centroidX - 1);
        int maxDistY = Math.min((int) centroidY, imgHeight - (int) centroidY - 1);
        int maxDistZ = Math.min((int) centroidZ, imgDepth - (int) centroidZ - 1);

        // Take minimum to ensure cubic region fits
        int optimalSize = Math.min(Math.min(maxDistX, maxDistY), maxDistZ) * 2;
        optimalSize = Math.min(optimalSize, maxSize);

        // Round down to nearest power of 2 for better compatibility with CNNs
        int size = 1;
        while (size * 2 <= optimalSize) {
            size *= 2;
        }

        return new int[]{size, size, size};
    }

    // ===== INSTANCE-BASED API (for VAE workflows) =====

    /**
     * Creates a CellRegionExtractor with fixed region size and padding type.
     * This constructor enables the instance-based API used by VAE workflows.
     *
     * @param regionSize The size of the cubic region (e.g., 64 for 64³)
     * @param paddingType The padding strategy for boundary regions
     */
    public CellRegionExtractor(int regionSize, PaddingType paddingType) {
        this.regionSize = regionSize;
        this.paddingType = paddingType;
    }

    /**
     * Extract a 3D cubic region around a MicroObject from a single-channel ImageStack.
     * Convenience method for VAE workflows that operate on single-channel data.
     *
     * @param cell MicroObject to extract region around
     * @param imageStack Single-channel ImageStack
     * @return Extracted region as ImageStack
     */
    public ImageStack extractRegion(MicroObject cell, ImageStack imageStack) {
        ImageStack[] result = extractRegion(
            cell,
            new ImageStack[]{imageStack},
            new int[]{regionSize, regionSize, regionSize},
            null,
            paddingType
        );
        return result[0];
    }

    /**
     * Extract 3D cubic regions from multi-channel ImageStacks using instance configuration.
     *
     * @param cell MicroObject to extract region around
     * @param imageStacks Array of ImageStacks (one per channel)
     * @return Array of extracted regions (one per channel)
     */
    public ImageStack[] extractRegion(MicroObject cell, ImageStack[] imageStacks) {
        return extractRegion(
            cell,
            imageStacks,
            new int[]{regionSize, regionSize, regionSize},
            null,
            paddingType
        );
    }
}
