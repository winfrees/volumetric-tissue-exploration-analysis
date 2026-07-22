/*
 * Copyright (C) 2026 Indiana University
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
package vtea.tools;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.plugin.ChannelSplitter;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import vtea.clustering.KMeans;
import vtea.objects.Segmentation.SingleThreshold;
import vtea.objects.measurements.Count;
import vtea.objects.measurements.Maximum;
import vtea.objects.measurements.Mean;
import vtea.objects.measurements.Minimum;
import vtea.objects.measurements.StandardDeviation;
import vtea.objects.measurements.Sum;
import vtea.reduction.PCAReduction;
import vteaobjects.MicroObject;

/**
 * Generates the golden-dataset parity fixtures used by the vtea-python port
 * (see PYTHON_PORT_PLAN.md Phase 0 / vtea-python/tests/golden). Runs the
 * real Java algorithm implementations (SingleThreshold segmentation, the
 * per-object measurement plugins, KMeans clustering, PCA reduction) against
 * fixed, reproducible inputs and writes CSV/TIFF outputs that the Python
 * port's test_parity.py diffs its own outputs against.
 *
 * Cannot be run in a network-restricted environment: the Maven build needs
 * maven.scijava.org for its dependencies. Intended to run in CI (see
 * .github/workflows/generate-golden-fixtures.yml) or on a developer machine
 * with normal internet access.
 *
 * Usage: mvn -q exec:java -Dexec.mainClass=vtea.tools.GoldenFixtureGenerator \
 *            -Dexec.args="<repoRoot> <outputDir>"
 * Both arguments are optional; default to "." and "target/golden-fixtures".
 */
public class GoldenFixtureGenerator {

    public static void main(String[] args) throws IOException {
        System.setProperty("java.awt.headless", "true");

        Path repoRoot = Paths.get(args.length > 0 ? args[0] : ".").toAbsolutePath();
        Path outDir = Paths.get(args.length > 1 ? args[1] : "target/golden-fixtures").toAbsolutePath();
        Files.createDirectories(outDir);

        String[] datasets = {"AQtest_human_crop.tif", "C1-IU_VTEA_ExampleData_001.tif"};
        for (String name : datasets) {
            Path tif = repoRoot.resolve(name);
            if (!Files.exists(tif)) {
                System.err.println("Skipping missing dataset: " + tif);
                continue;
            }
            generateImageDerivedFixture(tif, name, outDir);
        }

        generateSyntheticClusteringFixture(outDir);

        System.out.println("Golden fixtures written to " + outDir);
    }

    /**
     * Segments a real sample image with SingleThreshold3D and extracts
     * per-object intensity measurements. SingleThreshold produces exactly
     * one object (all above-threshold voxels) - this exercises image I/O,
     * thresholding, and measurement extraction end-to-end, but is not a
     * multi-object fixture. Richer multi-object segmentation fixtures
     * should be added once the corresponding methods are ported in Phase 2.
     */
    private static void generateImageDerivedFixture(Path tifPath, String datasetName, Path outDir) throws IOException {
        ImagePlus imp = IJ.openImage(tifPath.toString());
        if (imp == null) {
            throw new IOException("Failed to open " + tifPath);
        }

        int nChannels = imp.getNChannels();
        ImageStack[] stacks;
        if (nChannels <= 1) {
            stacks = new ImageStack[]{imp.getStack()};
        } else {
            ImagePlus[] split = ChannelSplitter.split(imp);
            stacks = new ImageStack[split.length];
            for (int c = 0; c < split.length; c++) {
                stacks[c] = split[c].getStack();
            }
        }

        int width = stacks[0].getWidth();
        int height = stacks[0].getHeight();
        int depth = stacks[0].getSize();

        double threshold = computeMeanPlus2StdDev(stacks[0]);

        SingleThreshold segmentation = new SingleThreshold();
        List<Object> protocol = buildSingleThresholdProtocol(threshold, 0);
        boolean ok = segmentation.process(stacks, protocol, true);
        if (!ok) {
            throw new RuntimeException("Segmentation failed for " + datasetName);
        }

        ArrayList<MicroObject> objects = segmentation.getObjects();
        String baseName = stripExtension(datasetName);

        writeLabelMask(objects, width, height, depth, outDir.resolve(baseName + "_segmentation_singlethreshold.tif"), datasetName);
        writeMeasurements(objects, stacks[0], outDir.resolve(baseName + "_measurements.csv"));

        String meta = "dataset=" + datasetName + "\n"
                + "width=" + width + "\nheight=" + height + "\ndepth=" + depth + "\nchannels=" + nChannels + "\n"
                + "segmentation_method=SingleThreshold3D\n"
                + "threshold=" + threshold + " (mean + 2*stddev of channel 0)\n"
                + "object_count=" + objects.size() + "\n";
        Files.writeString(outDir.resolve(baseName + "_metadata.txt"), meta);

        System.out.println("Wrote fixtures for " + datasetName + ": " + objects.size() + " object(s), threshold=" + threshold);
    }

    private static void writeLabelMask(ArrayList<MicroObject> objects, int width, int height, int depth,
                                        Path outPath, String title) throws IOException {
        ImageStack labelStack = new ImageStack(width, height);
        for (int z = 0; z < depth; z++) {
            labelStack.addSlice(new ShortProcessor(width, height));
        }
        for (int i = 0; i < objects.size(); i++) {
            MicroObject obj = objects.get(i);
            int[] xs = obj.getPixelsX();
            int[] ys = obj.getPixelsY();
            int[] zs = obj.getPixelsZ();
            short label = (short) (i + 1);
            for (int p = 0; p < xs.length; p++) {
                ShortProcessor sp = (ShortProcessor) labelStack.getProcessor(zs[p] + 1);
                sp.set(xs[p], ys[p], label);
            }
        }
        ImagePlus labelImp = new ImagePlus(title + "_labels", labelStack);
        new FileSaver(labelImp).saveAsTiff(outPath.toString());
    }

    private static void writeMeasurements(ArrayList<MicroObject> objects, ImageStack channel0, Path outPath) throws IOException {
        StringBuilder csv = new StringBuilder("object_id,count,mean,sum,stddev,min,max\n");
        Mean meanOp = new Mean();
        Sum sumOp = new Sum();
        StandardDeviation stdOp = new StandardDeviation();
        Count countOp = new Count();
        Minimum minOp = new Minimum();
        Maximum maxOp = new Maximum();

        for (int i = 0; i < objects.size(); i++) {
            MicroObject obj = objects.get(i);
            int[] xs = obj.getPixelsX();
            int[] ys = obj.getPixelsY();
            int[] zs = obj.getPixelsZ();
            ArrayList<Number> values = new ArrayList<>(xs.length);
            for (int p = 0; p < xs.length; p++) {
                values.add(channel0.getProcessor(zs[p] + 1).getPixelValue(xs[p], ys[p]));
            }
            csv.append(i + 1).append(',')
                    .append(countOp.process(null, values)).append(',')
                    .append(meanOp.process(null, values)).append(',')
                    .append(sumOp.process(null, values)).append(',')
                    .append(stdOp.process(null, values)).append(',')
                    .append(minOp.process(null, values)).append(',')
                    .append(maxOp.process(null, values)).append('\n');
        }
        Files.writeString(outPath, csv.toString());
    }

    private static double computeMeanPlus2StdDev(ImageStack stack) {
        long count = 0;
        double sum = 0;
        int depth = stack.getSize();
        int width = stack.getWidth();
        int height = stack.getHeight();
        for (int z = 1; z <= depth; z++) {
            ImageProcessor ip = stack.getProcessor(z);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    sum += ip.getPixelValue(x, y);
                    count++;
                }
            }
        }
        double mean = sum / count;
        double sqSum = 0;
        for (int z = 1; z <= depth; z++) {
            ImageProcessor ip = stack.getProcessor(z);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double d = ip.getPixelValue(x, y) - mean;
                    sqSum += d * d;
                }
            }
        }
        double stddev = Math.sqrt(sqSum / count);
        return mean + 2 * stddev;
    }

    private static List<Object> buildSingleThresholdProtocol(double threshold, int channel) {
        List<Object> protocol = new ArrayList<>();
        protocol.add("Golden Fixture Segmentation");
        protocol.add("SingleThreshold3D");
        protocol.add(channel);
        ArrayList<Object> components = new ArrayList<>();
        components.add(new JLabel("Low Threshold"));
        components.add(new JTextField(String.valueOf((int) threshold), 5));
        protocol.add(components);
        return protocol;
    }

    /**
     * Deterministic synthetic data (seed=42, 300 points / 2D / 3 clusters),
     * matching the pattern in src/test/java/vtea/TestDataBuilder, run
     * through the real KMeans and PCA implementations. Independent of image
     * segmentation, so it isolates clustering/DR algorithmic parity.
     */
    @SuppressWarnings("unchecked")
    private static void generateSyntheticClusteringFixture(Path outDir) throws IOException {
        Random random = new Random(42);
        int numPoints = 300, numDimensions = 2, numClusters = 3;
        double clusterStdDev = 5.0, clusterSeparation = 50.0;
        double[][] data = new double[numPoints][numDimensions];
        double[][] centers = new double[numClusters][numDimensions];
        for (int i = 0; i < numClusters; i++) {
            for (int j = 0; j < numDimensions; j++) {
                centers[i][j] = i * clusterSeparation + random.nextDouble() * 10;
            }
        }
        int pointsPerCluster = numPoints / numClusters;
        int pointIndex = 0;
        for (int cluster = 0; cluster < numClusters; cluster++) {
            int pointsInThisCluster = (cluster == numClusters - 1) ? numPoints - pointIndex : pointsPerCluster;
            for (int p = 0; p < pointsInThisCluster; p++) {
                for (int dim = 0; dim < numDimensions; dim++) {
                    data[pointIndex][dim] = centers[cluster][dim] + random.nextGaussian() * clusterStdDev;
                }
                pointIndex++;
            }
        }

        StringBuilder inputCsv = new StringBuilder("point_id,x,y\n");
        for (int i = 0; i < data.length; i++) {
            inputCsv.append(i).append(',').append(data[i][0]).append(',').append(data[i][1]).append('\n');
        }
        Files.writeString(outDir.resolve("synthetic_clustering_input.csv"), inputCsv.toString());

        ArrayList<Integer> selectData = new ArrayList<>();
        selectData.add(0);
        selectData.add(1);

        KMeans kmeans = new KMeans();
        ArrayList<Object> kProtocol = new ArrayList<>();
        kProtocol.add(false); // z-normalization off
        kProtocol.add(selectData);
        kProtocol.add(null);
        kProtocol.add(null);
        kProtocol.add(new JLabel("Clusters"));
        kProtocol.add(new JSpinner(new SpinnerNumberModel(numClusters, 2, 100, 1)));
        kProtocol.add(new JLabel("Iterations"));
        kProtocol.add(new JTextField("50"));
        kmeans.process(kProtocol, data, false);
        ArrayList<Object> membership = (ArrayList<Object>) kmeans.getResult().get(0);

        StringBuilder kmeansCsv = new StringBuilder("point_id,cluster\n");
        for (int i = 0; i < membership.size(); i++) {
            kmeansCsv.append(i).append(',').append(membership.get(i)).append('\n');
        }
        Files.writeString(outDir.resolve("synthetic_clustering_kmeans_k3.csv"), kmeansCsv.toString());

        PCAReduction pca = new PCAReduction(numPoints);
        ArrayList<Object> pcaProtocol = new ArrayList<>();
        pcaProtocol.add(false);
        pcaProtocol.add(selectData);
        pcaProtocol.add(null);
        pcaProtocol.add(null);
        pcaProtocol.add(new JComboBox<String>(new String[]{"New Dimension", "Desired Variance"}));
        pcaProtocol.add(new JTextField("2"));
        pca.process(pcaProtocol, data, false);
        ArrayList<Object> result = pca.getResult();

        StringBuilder pcaCsv = new StringBuilder("point_id");
        for (int d = 0; d < result.size(); d++) {
            pcaCsv.append(",pc").append(d + 1);
        }
        pcaCsv.append('\n');
        int nRows = ((ArrayList<?>) result.get(0)).size();
        for (int i = 0; i < nRows; i++) {
            pcaCsv.append(i);
            for (int d = 0; d < result.size(); d++) {
                pcaCsv.append(',').append(((ArrayList<?>) result.get(d)).get(i));
            }
            pcaCsv.append('\n');
        }
        Files.writeString(outDir.resolve("synthetic_clustering_pca.csv"), pcaCsv.toString());

        System.out.println("Wrote synthetic clustering/PCA fixtures (" + numPoints + " points, k=" + numClusters + ")");
    }

    private static String stripExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }
}
