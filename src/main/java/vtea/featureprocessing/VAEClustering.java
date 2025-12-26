/*
 * Copyright (C) 2025 Indiana University
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
package vtea.featureprocessing;

import ij.ImageStack;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import org.bytedeco.pytorch.Tensor;
import org.scijava.plugin.Plugin;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import vtea.deeplearning.CellRegionExtractor;
import vtea.deeplearning.TensorConverter;
import vtea.deeplearning.models.VariationalAutoencoder3D;
import vtea.objects.MicroObject;
import vtea.protocol.setup.MicroBlockSetup;

/**
 * K-means clustering in 3D VAE latent space.
 * Groups cells based on their learned latent representations.
 *
 * @author VTEA Developer
 */
@Plugin(type = FeatureProcessing.class)
public class VAEClustering extends AbstractFeatureProcessing {

    private static final Logger logger = LoggerFactory.getLogger(VAEClustering.class);

    private VariationalAutoencoder3D model;
    private CellRegionExtractor extractor;
    private TensorConverter converter;
    private String modelPath;
    private int numClusters;
    private int maxIterations;
    private Random random;

    /**
     * Creates the Comment Text for the Block GUI.
     *
     * @param comComponents the parameters (Components) selected by the user
     * @return comment text detailing the parameters
     */
    public static String getBlockComment(ArrayList comComponents) {
        String comment = "<html>";
        comment = comment.concat("VAE K-means Clustering<br>");

        JTextField modelField = (JTextField) comComponents.get(1);
        String path = modelField.getText();
        if (path != null && !path.isEmpty()) {
            File f = new File(path);
            comment = comment.concat("Model: " + f.getName() + "<br>");
        } else {
            comment = comment.concat("Model: Not selected<br>");
        }

        JSpinner clusterSpinner = (JSpinner) comComponents.get(4);
        comment = comment.concat("Clusters: " + clusterSpinner.getValue() + "<br>");

        JTextField iterField = (JTextField) comComponents.get(6);
        comment = comment.concat("Iterations: " + iterField.getText());

        comment = comment.concat("</html>");
        return comment;
    }

    /**
     * Basic Constructor. Sets all protected variables.
     */
    public VAEClustering() {
        VERSION = "1.0";
        AUTHOR = "VTEA Developer";
        COMMENT = "K-means clustering in VAE latent space";
        NAME = "VAE Clustering";
        KEY = "VAEClustering";
        TYPE = "Cluster";
        random = new Random(42);
    }

    /**
     * Constructor called for initialization of Setup GUI.
     *
     * @param max the number of objects segmented in the volume
     */
    public VAEClustering(int max) {
        this();

        protocol = new ArrayList();

        // Model path selection
        protocol.add(new JLabel("VAE Model Path:"));
        JTextField modelPathField = new JTextField(30);
        protocol.add(modelPathField);

        JButton browseButton = new JButton("Browse...");
        browseButton.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser();
            chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            chooser.setDialogTitle("Select VAE Model Checkpoint Directory");

            int result = chooser.showOpenDialog(null);
            if (result == JFileChooser.APPROVE_OPTION) {
                File selectedDir = chooser.getSelectedFile();
                modelPathField.setText(selectedDir.getAbsolutePath());
            }
        });
        protocol.add(browseButton);

        // Number of clusters
        protocol.add(new JLabel("Number of Clusters:"));
        protocol.add(new JSpinner(new SpinnerNumberModel(5, 2, max, 1)));

        // Max iterations
        protocol.add(new JLabel("Max Iterations:"));
        protocol.add(new JTextField("100", 5));
    }

    @Override
    public String getDataDescription(ArrayList params) {
        JSpinner clusterSpinner = (JSpinner) params.get(4);
        return "VAE_Cluster_K" + clusterSpinner.getValue();
    }

    /**
     * Performs k-means clustering on latent features.
     *
     * @param data Input data [n_samples][n_features]
     * @param k Number of clusters
     * @param maxIter Maximum iterations
     * @return Cluster assignments [n_samples]
     */
    private int[] performKMeans(double[][] data, int k, int maxIter) {
        int n = data.length;
        int d = data[0].length;

        // Initialize centroids randomly
        double[][] centroids = new double[k][d];
        boolean[] selected = new boolean[n];
        for (int i = 0; i < k; i++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (selected[idx]);
            selected[idx] = true;
            System.arraycopy(data[idx], 0, centroids[i], 0, d);
        }

        int[] assignments = new int[n];
        boolean changed = true;
        int iteration = 0;

        while (changed && iteration < maxIter) {
            changed = false;

            // Assign points to nearest centroid
            for (int i = 0; i < n; i++) {
                int nearestCluster = 0;
                double minDistance = Double.MAX_VALUE;

                for (int j = 0; j < k; j++) {
                    double distance = 0.0;
                    for (int l = 0; l < d; l++) {
                        double diff = data[i][l] - centroids[j][l];
                        distance += diff * diff;
                    }

                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCluster = j;
                    }
                }

                if (assignments[i] != nearestCluster) {
                    assignments[i] = nearestCluster;
                    changed = true;
                }
            }

            // Update centroids
            int[] counts = new int[k];
            double[][] newCentroids = new double[k][d];

            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                counts[cluster]++;
                for (int j = 0; j < d; j++) {
                    newCentroids[cluster][j] += data[i][j];
                }
            }

            for (int i = 0; i < k; i++) {
                if (counts[i] > 0) {
                    for (int j = 0; j < d; j++) {
                        centroids[i][j] = newCentroids[i][j] / counts[i];
                    }
                }
            }

            iteration++;
            logger.debug("K-means iteration {}/{}", iteration, maxIter);
        }

        logger.info("K-means converged after {} iterations", iteration);
        return assignments;
    }

    /**
     * Processes the clustering using trained VAE model.
     *
     * @param al ArrayList containing GUI components with parameters
     * @param feature 2D array of existing features [objects][features]
     * @param validate whether to perform validation
     * @return true if successful, false otherwise
     */
    @Override
    public boolean process(ArrayList al, double[][] feature, boolean validate) {
        logger.info("Starting VAE clustering...");

        try {
            // Extract parameters from GUI components
            JTextField modelPathField = (JTextField) al.get(1);
            modelPath = modelPathField.getText();

            JSpinner clusterSpinner = (JSpinner) al.get(4);
            numClusters = (int) clusterSpinner.getValue();

            JTextField iterField = (JTextField) al.get(6);
            maxIterations = Integer.parseInt(iterField.getText());

            if (modelPath == null || modelPath.isEmpty()) {
                logger.error("Model path not specified");
                return false;
            }

            File modelDir = new File(modelPath);
            if (!modelDir.exists() || !modelDir.isDirectory()) {
                logger.error("Model directory does not exist: {}", modelPath);
                return false;
            }

            // Load the trained VAE model
            logger.info("Loading VAE model from: {}", modelPath);
            model = VariationalAutoencoder3D.load(modelPath);
            model.eval();

            // Get config to determine input size
            int inputSize = model.getConfig().getInputSize();
            int latentDim = model.getConfig().getLatentDim();
            logger.info("Model input size: {}^3, latent dim: {}", inputSize, latentDim);

            // Initialize extractors
            extractor = new CellRegionExtractor(inputSize, CellRegionExtractor.PaddingType.REPLICATE);
            converter = new TensorConverter(
                TensorConverter.NormalizationType.ZSCORE,
                model.getConfig().isUseGPU()
            );

            // Get image data and objects from MicroBlockSetup
            ArrayList<MicroObject> objects = MicroBlockSetup.getMicroObjects();
            ImageStack imageStack = MicroBlockSetup.getImageStack();

            if (objects == null || objects.isEmpty()) {
                logger.error("No objects found in MicroBlockSetup");
                return false;
            }

            if (imageStack == null) {
                logger.error("No image stack found in MicroBlockSetup");
                return false;
            }

            logger.info("Clustering {} objects into {} clusters", objects.size(), numClusters);

            // Extract latent features for all objects
            double[][] latentFeatures = new double[objects.size()][latentDim];

            for (int i = 0; i < objects.size(); i++) {
                MicroObject cell = objects.get(i);

                // Extract 3D region around cell
                ImageStack region = extractor.extractRegion(cell, imageStack);

                // Convert to tensor
                Tensor inputTensor = converter.imageStackToTensor(region);

                // Encode to latent space
                Tensor latentTensor = model.encode(inputTensor);

                // Extract latent values
                for (int j = 0; j < latentDim; j++) {
                    latentFeatures[i][j] = latentTensor.index_get(j).item().doubleValue();
                }

                // Cleanup
                inputTensor.close();
                latentTensor.close();

                // Progress tracking
                if (i % 100 == 0) {
                    progress = (int) ((i / (double) objects.size()) * 50);
                    logger.debug("Extracted latent features for {}/{} objects", i, objects.size());
                }
            }

            // Perform k-means clustering
            logger.info("Running k-means clustering with k={}, max_iter={}", numClusters, maxIterations);
            int[] clusterAssignments = performKMeans(latentFeatures, numClusters, maxIterations);

            // Store results
            dataResult = new ArrayList();
            for (int i = 0; i < clusterAssignments.length; i++) {
                dataResult.add(clusterAssignments[i]);
            }

            // Log cluster distribution
            int[] clusterCounts = new int[numClusters];
            for (int assignment : clusterAssignments) {
                clusterCounts[assignment]++;
            }
            StringBuilder distribution = new StringBuilder("Cluster distribution: ");
            for (int i = 0; i < numClusters; i++) {
                distribution.append(String.format("C%d=%d ", i, clusterCounts[i]));
            }
            logger.info(distribution.toString());

            progress = 100;
            logger.info("VAE clustering complete. Assigned {} objects to {} clusters",
                       objects.size(), numClusters);

            return true;

        } catch (Exception e) {
            logger.error("Error during VAE clustering", e);
            return false;
        } finally {
            // Cleanup resources
            if (model != null) {
                model.close();
            }
        }
    }
}
