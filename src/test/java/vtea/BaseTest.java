/*
 * Copyright (C) 2020 Indiana University
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
package vtea;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import java.util.ArrayList;
import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

/**
 * Base test class providing common utilities for VTEA tests.
 *
 * All test classes should extend this class to access common test utilities,
 * mock data generators, and assertion helpers.
 *
 * @author VTEA Development Team
 */
public abstract class BaseTest {

    protected Random random;
    protected static final double DELTA = 0.001; // Default delta for float comparisons

    @BeforeEach
    public void baseSetUp() {
        // Initialize with fixed seed for reproducible tests
        random = new Random(42);
    }

    @AfterEach
    public void baseTearDown() {
        // Cleanup if needed
        random = null;
    }

    // ========== Image Creation Utilities ==========

    /**
     * Creates a simple 8-bit test image with specified dimensions.
     *
     * @param width image width
     * @param height image height
     * @return ImagePlus with random pixel values
     */
    protected ImagePlus createTestImage8bit(int width, int height) {
        ByteProcessor processor = new ByteProcessor(width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                processor.set(x, y, random.nextInt(256));
            }
        }
        return new ImagePlus("Test8bit", processor);
    }

    /**
     * Creates a 16-bit test image with specified dimensions.
     *
     * @param width image width
     * @param height image height
     * @return ImagePlus with random pixel values
     */
    protected ImagePlus createTestImage16bit(int width, int height) {
        ShortProcessor processor = new ShortProcessor(width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                processor.set(x, y, random.nextInt(65536));
            }
        }
        return new ImagePlus("Test16bit", processor);
    }

    /**
     * Creates a 32-bit float test image with specified dimensions.
     *
     * @param width image width
     * @param height image height
     * @return ImagePlus with random pixel values
     */
    protected ImagePlus createTestImage32bit(int width, int height) {
        FloatProcessor processor = new FloatProcessor(width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                processor.setf(x, y, random.nextFloat() * 1000);
            }
        }
        return new ImagePlus("Test32bit", processor);
    }

    /**
     * Creates a 3D image stack with specified dimensions.
     *
     * @param width image width
     * @param height image height
     * @param depth number of slices
     * @return ImagePlus with 3D stack
     */
    protected ImagePlus createTestStack(int width, int height, int depth) {
        ImageStack stack = new ImageStack(width, height);
        for (int z = 0; z < depth; z++) {
            ByteProcessor processor = new ByteProcessor(width, height);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    processor.set(x, y, random.nextInt(256));
                }
            }
            stack.addSlice("Slice " + (z + 1), processor);
        }
        ImagePlus imp = new ImagePlus("TestStack", stack);
        imp.setDimensions(1, depth, 1); // 1 channel, depth slices, 1 timepoint
        return imp;
    }

    /**
     * Creates a synthetic image with a circular object.
     *
     * @param width image width
     * @param height image height
     * @param centerX circle center X
     * @param centerY circle center Y
     * @param radius circle radius
     * @param foreground foreground intensity
     * @param background background intensity
     * @return ImagePlus with circular object
     */
    protected ImagePlus createCircularObject(int width, int height, int centerX,
                                             int centerY, int radius,
                                             int foreground, int background) {
        ByteProcessor processor = new ByteProcessor(width, height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
                if (distance <= radius) {
                    processor.set(x, y, foreground);
                } else {
                    processor.set(x, y, background);
                }
            }
        }

        return new ImagePlus("CircularObject", processor);
    }

    /**
     * Creates a synthetic image with a rectangular object.
     *
     * @param width image width
     * @param height image height
     * @param rectX rectangle top-left X
     * @param rectY rectangle top-left Y
     * @param rectWidth rectangle width
     * @param rectHeight rectangle height
     * @param foreground foreground intensity
     * @param background background intensity
     * @return ImagePlus with rectangular object
     */
    protected ImagePlus createRectangularObject(int width, int height,
                                                int rectX, int rectY,
                                                int rectWidth, int rectHeight,
                                                int foreground, int background) {
        ByteProcessor processor = new ByteProcessor(width, height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (x >= rectX && x < rectX + rectWidth &&
                    y >= rectY && y < rectY + rectHeight) {
                    processor.set(x, y, foreground);
                } else {
                    processor.set(x, y, background);
                }
            }
        }

        return new ImagePlus("RectangularObject", processor);
    }

    // ========== Pixel List Utilities ==========

    /**
     * Creates a pixel list for a single pixel.
     *
     * @param x x coordinate
     * @param y y coordinate
     * @param z z coordinate
     * @return ArrayList containing single pixel
     */
    protected ArrayList<int[]> createSinglePixel(int x, int y, int z) {
        ArrayList<int[]> pixels = new ArrayList<>();
        pixels.add(new int[]{x, y, z});
        return pixels;
    }

    /**
     * Creates a pixel list for a line of pixels.
     *
     * @param x1 start X
     * @param y1 start Y
     * @param x2 end X
     * @param y2 end Y
     * @param z z coordinate
     * @return ArrayList containing line pixels
     */
    protected ArrayList<int[]> createLinePixels(int x1, int y1, int x2, int y2, int z) {
        ArrayList<int[]> pixels = new ArrayList<>();

        int dx = Math.abs(x2 - x1);
        int dy = Math.abs(y2 - y1);
        int sx = x1 < x2 ? 1 : -1;
        int sy = y1 < y2 ? 1 : -1;
        int err = dx - dy;

        int x = x1;
        int y = y1;

        while (true) {
            pixels.add(new int[]{x, y, z});

            if (x == x2 && y == y2) break;

            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
        }

        return pixels;
    }

    /**
     * Creates a pixel list for a square region.
     *
     * @param centerX center X
     * @param centerY center Y
     * @param z z coordinate
     * @param size square size (pixels from center)
     * @return ArrayList containing square pixels
     */
    protected ArrayList<int[]> createSquarePixels(int centerX, int centerY, int z, int size) {
        ArrayList<int[]> pixels = new ArrayList<>();

        for (int y = centerY - size; y <= centerY + size; y++) {
            for (int x = centerX - size; x <= centerX + size; x++) {
                pixels.add(new int[]{x, y, z});
            }
        }

        return pixels;
    }

    // ========== Data Generation Utilities ==========

    /**
     * Creates a 2D array with random values.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @return 2D double array
     */
    protected double[][] createRandomData(int rows, int cols) {
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = random.nextDouble() * 100;
            }
        }
        return data;
    }

    /**
     * Creates a 2D array with Gaussian clusters.
     *
     * @param pointsPerCluster number of points per cluster
     * @param numClusters number of clusters
     * @param dimensions number of dimensions
     * @return 2D array where each row is a point
     */
    protected double[][] createGaussianClusters(int pointsPerCluster, int numClusters, int dimensions) {
        double[][] data = new double[pointsPerCluster * numClusters][dimensions];

        // Generate cluster centers
        double[][] centers = new double[numClusters][dimensions];
        for (int i = 0; i < numClusters; i++) {
            for (int j = 0; j < dimensions; j++) {
                centers[i][j] = random.nextDouble() * 100;
            }
        }

        // Generate points around each center
        int pointIndex = 0;
        for (int cluster = 0; cluster < numClusters; cluster++) {
            for (int point = 0; point < pointsPerCluster; point++) {
                for (int dim = 0; dim < dimensions; dim++) {
                    // Add Gaussian noise around center
                    data[pointIndex][dim] = centers[cluster][dim] + random.nextGaussian() * 5;
                }
                pointIndex++;
            }
        }

        return data;
    }

    // ========== Assertion Helpers ==========

    /**
     * Checks if two double values are approximately equal.
     *
     * @param expected expected value
     * @param actual actual value
     * @param delta tolerance
     * @return true if within tolerance
     */
    protected boolean approximatelyEqual(double expected, double actual, double delta) {
        return Math.abs(expected - actual) < delta;
    }

    /**
     * Checks if an array contains a specific value.
     *
     * @param array array to search
     * @param value value to find
     * @return true if found
     */
    protected boolean arrayContains(int[] array, int value) {
        for (int v : array) {
            if (v == value) return true;
        }
        return false;
    }
}
