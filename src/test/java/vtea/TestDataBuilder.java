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
import ij.process.ImageProcessor;
import java.util.ArrayList;
import java.util.Random;

/**
 * Builder class for creating complex test data objects.
 *
 * Uses the Builder pattern to make test data creation more readable
 * and flexible.
 *
 * @author VTEA Development Team
 */
public class TestDataBuilder {

    private static final Random RANDOM = new Random(42);

    /**
     * Builder for creating test ImagePlus objects.
     */
    public static class ImageBuilder {
        private String title = "TestImage";
        private int width = 512;
        private int height = 512;
        private int depth = 1;
        private int channels = 1;
        private int bitDepth = 8;
        private boolean addNoise = false;
        private ArrayList<SyntheticObject> objects = new ArrayList<>();

        public ImageBuilder title(String title) {
            this.title = title;
            return this;
        }

        public ImageBuilder size(int width, int height) {
            this.width = width;
            this.height = height;
            return this;
        }

        public ImageBuilder depth(int depth) {
            this.depth = depth;
            return this;
        }

        public ImageBuilder channels(int channels) {
            this.channels = channels;
            return this;
        }

        public ImageBuilder bitDepth(int bitDepth) {
            this.bitDepth = bitDepth;
            return this;
        }

        public ImageBuilder withNoise() {
            this.addNoise = true;
            return this;
        }

        public ImageBuilder addCircle(int x, int y, int z, int radius, int intensity) {
            objects.add(new SyntheticObject(SyntheticObject.Type.CIRCLE,
                    x, y, z, radius, radius, intensity));
            return this;
        }

        public ImageBuilder addRectangle(int x, int y, int z, int width, int height, int intensity) {
            objects.add(new SyntheticObject(SyntheticObject.Type.RECTANGLE,
                    x, y, z, width, height, intensity));
            return this;
        }

        public ImagePlus build() {
            if (depth == 1) {
                return buildSingleImage();
            } else {
                return buildStack();
            }
        }

        private ImagePlus buildSingleImage() {
            ByteProcessor processor = new ByteProcessor(width, height);

            // Add background
            if (addNoise) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        processor.set(x, y, RANDOM.nextInt(50)); // Low intensity noise
                    }
                }
            }

            // Add objects
            for (SyntheticObject obj : objects) {
                if (obj.z == 0) {
                    obj.draw(processor);
                }
            }

            return new ImagePlus(title, processor);
        }

        private ImagePlus buildStack() {
            ImageStack stack = new ImageStack(width, height);

            for (int z = 0; z < depth; z++) {
                ByteProcessor processor = new ByteProcessor(width, height);

                // Add background
                if (addNoise) {
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            processor.set(x, y, RANDOM.nextInt(50));
                        }
                    }
                }

                // Add objects on this slice
                final int sliceZ = z;
                for (SyntheticObject obj : objects) {
                    if (obj.z == sliceZ) {
                        obj.draw(processor);
                    }
                }

                stack.addSlice("Slice " + (z + 1), processor);
            }

            ImagePlus imp = new ImagePlus(title, stack);
            imp.setDimensions(channels, depth, 1);
            return imp;
        }
    }

    /**
     * Represents a synthetic object to draw in test images.
     */
    private static class SyntheticObject {
        enum Type { CIRCLE, RECTANGLE }

        Type type;
        int x, y, z;
        int width, height;
        int intensity;

        SyntheticObject(Type type, int x, int y, int z, int width, int height, int intensity) {
            this.type = type;
            this.x = x;
            this.y = y;
            this.z = z;
            this.width = width;
            this.height = height;
            this.intensity = intensity;
        }

        void draw(ImageProcessor processor) {
            switch (type) {
                case CIRCLE:
                    drawCircle(processor);
                    break;
                case RECTANGLE:
                    drawRectangle(processor);
                    break;
            }
        }

        void drawCircle(ImageProcessor processor) {
            int radius = width;
            for (int py = -radius; py <= radius; py++) {
                for (int px = -radius; px <= radius; px++) {
                    if (px * px + py * py <= radius * radius) {
                        int drawX = x + px;
                        int drawY = y + py;
                        if (drawX >= 0 && drawX < processor.getWidth() &&
                            drawY >= 0 && drawY < processor.getHeight()) {
                            processor.set(drawX, drawY, intensity);
                        }
                    }
                }
            }
        }

        void drawRectangle(ImageProcessor processor) {
            for (int py = 0; py < height; py++) {
                for (int px = 0; px < width; px++) {
                    int drawX = x + px;
                    int drawY = y + py;
                    if (drawX >= 0 && drawX < processor.getWidth() &&
                        drawY >= 0 && drawY < processor.getHeight()) {
                        processor.set(drawX, drawY, intensity);
                    }
                }
            }
        }
    }

    /**
     * Builder for creating test feature data.
     */
    public static class FeatureDataBuilder {
        private int numPoints = 100;
        private int numDimensions = 2;
        private int numClusters = 3;
        private double clusterStdDev = 5.0;
        private double clusterSeparation = 50.0;

        public FeatureDataBuilder points(int numPoints) {
            this.numPoints = numPoints;
            return this;
        }

        public FeatureDataBuilder dimensions(int numDimensions) {
            this.numDimensions = numDimensions;
            return this;
        }

        public FeatureDataBuilder clusters(int numClusters) {
            this.numClusters = numClusters;
            return this;
        }

        public FeatureDataBuilder clusterStdDev(double stdDev) {
            this.clusterStdDev = stdDev;
            return this;
        }

        public FeatureDataBuilder clusterSeparation(double separation) {
            this.clusterSeparation = separation;
            return this;
        }

        public double[][] build() {
            double[][] data = new double[numPoints][numDimensions];
            int pointsPerCluster = numPoints / numClusters;

            // Generate cluster centers
            double[][] centers = new double[numClusters][numDimensions];
            for (int i = 0; i < numClusters; i++) {
                for (int j = 0; j < numDimensions; j++) {
                    centers[i][j] = i * clusterSeparation + RANDOM.nextDouble() * 10;
                }
            }

            // Generate points around each center
            int pointIndex = 0;
            for (int cluster = 0; cluster < numClusters; cluster++) {
                int pointsInThisCluster = (cluster == numClusters - 1) ?
                        numPoints - pointIndex : pointsPerCluster;

                for (int point = 0; point < pointsInThisCluster; point++) {
                    for (int dim = 0; dim < numDimensions; dim++) {
                        data[pointIndex][dim] = centers[cluster][dim] +
                                RANDOM.nextGaussian() * clusterStdDev;
                    }
                    pointIndex++;
                }
            }

            return data;
        }
    }

    // Factory methods for convenience

    public static ImageBuilder image() {
        return new ImageBuilder();
    }

    public static FeatureDataBuilder features() {
        return new FeatureDataBuilder();
    }

    /**
     * Creates a simple 2D test image with default parameters.
     *
     * @return 512x512 8-bit image
     */
    public static ImagePlus createSimpleImage() {
        return image().build();
    }

    /**
     * Creates a test image with a single circular object.
     *
     * @return image with circle in center
     */
    public static ImagePlus createImageWithCircle() {
        return image()
                .size(256, 256)
                .addCircle(128, 128, 0, 50, 255)
                .build();
    }

    /**
     * Creates a test image with multiple objects.
     *
     * @return image with 3 circles
     */
    public static ImagePlus createImageWithMultipleObjects() {
        return image()
                .size(512, 512)
                .addCircle(128, 128, 0, 30, 255)
                .addCircle(384, 128, 0, 40, 255)
                .addCircle(256, 384, 0, 35, 255)
                .build();
    }

    /**
     * Creates a 3D stack with objects.
     *
     * @return 3D stack with objects on different slices
     */
    public static ImagePlus create3DStack() {
        return image()
                .size(256, 256)
                .depth(10)
                .addCircle(128, 128, 3, 40, 255)
                .addCircle(128, 128, 7, 40, 255)
                .build();
    }

    /**
     * Creates sample clustering data with well-separated clusters.
     *
     * @return 2D array of feature vectors
     */
    public static double[][] createClusteringData() {
        return features()
                .points(300)
                .dimensions(2)
                .clusters(3)
                .clusterStdDev(5.0)
                .clusterSeparation(50.0)
                .build();
    }
}
