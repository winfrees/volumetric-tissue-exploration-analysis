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
import java.util.Arrays;
import javax.swing.JButton;
import javax.swing.JComboBox;
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
 * Anomaly detection using 3D VAE reconstruction error.
 * Identifies cells with high reconstruction error as potential anomalies.
 *
 * @author VTEA Developer
 */
@Plugin(type = FeatureProcessing.class)
public class VAEAnomalyDetection extends AbstractFeatureProcessing {

    private static final Logger logger = LoggerFactory.getLogger(VAEAnomalyDetection.class);

    private VariationalAutoencoder3D model;
    private CellRegionExtractor extractor;
    private TensorConverter converter;
    private String modelPath;
    private String outputMode;
    private double threshold;

    /**
     * Creates the Comment Text for the Block GUI.
     *
     * @param comComponents the parameters (Components) selected by the user
     * @return comment text detailing the parameters
     */
    public static String getBlockComment(ArrayList comComponents) {
        String comment = "<html>";
        comment = comment.concat("VAE Anomaly Detection<br>");

        JTextField modelField = (JTextField) comComponents.get(1);
        String path = modelField.getText();
        if (path != null && !path.isEmpty()) {
            File f = new File(path);
            comment = comment.concat("Model: " + f.getName() + "<br>");
        } else {
            comment = comment.concat("Model: Not selected<br>");
        }

        JComboBox<?> modeCombo = (JComboBox<?>) comComponents.get(4);
        comment = comment.concat("Output: " + modeCombo.getSelectedItem() + "<br>");

        JSpinner thresholdSpinner = (JSpinner) comComponents.get(6);
        comment = comment.concat("Threshold: " + thresholdSpinner.getValue() + " SD");

        comment = comment.concat("</html>");
        return comment;
    }

    /**
     * Basic Constructor. Sets all protected variables.
     */
    public VAEAnomalyDetection() {
        VERSION = "1.0";
        AUTHOR = "VTEA Developer";
        COMMENT = "Detect anomalous cells using VAE reconstruction error";
        NAME = "VAE Anomaly Detection";
        KEY = "VAEAnomaly";
        TYPE = "Other";
    }

    /**
     * Constructor called for initialization of Setup GUI.
     *
     * @param max the number of objects segmented in the volume (unused)
     */
    public VAEAnomalyDetection(int max) {
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

        // Output mode selection
        protocol.add(new JLabel("Output Mode:"));
        String[] modes = {"Reconstruction Error", "Anomaly Score (0-1)", "Binary (Normal/Anomaly)"};
        JComboBox<String> modeCombo = new JComboBox<>(modes);
        modeCombo.setSelectedIndex(0); // Default to reconstruction error
        protocol.add(modeCombo);

        // Threshold for binary classification (in standard deviations)
        protocol.add(new JLabel("Anomaly Threshold (SD):"));
        protocol.add(new JSpinner(new SpinnerNumberModel(2.0, 1.0, 5.0, 0.5)));
    }

    @Override
    public String getDataDescription(ArrayList params) {
        JComboBox<?> modeCombo = (JComboBox<?>) params.get(4);
        String mode = (String) modeCombo.getSelectedItem();
        if (mode.contains("Binary")) {
            return "VAE_Anomaly_Binary";
        } else if (mode.contains("Score")) {
            return "VAE_Anomaly_Score";
        } else {
            return "VAE_Recon_Error";
        }
    }

    /**
     * Computes Mean Squared Error between two tensors.
     *
     * @param reconstruction Reconstructed tensor
     * @param original Original tensor
     * @return MSE value
     */
    private double computeMSE(Tensor reconstruction, Tensor original) {
        Tensor diff = reconstruction.sub(original);
        Tensor squared = diff.pow(2.0);
        double mse = squared.mean().item().doubleValue();

        diff.close();
        squared.close();

        return mse;
    }

    /**
     * Normalizes reconstruction errors to anomaly scores [0, 1].
     *
     * @param errors Array of reconstruction errors
     * @return Array of normalized scores (0 = normal, 1 = highly anomalous)
     */
    private double[] normalizeToScores(double[] errors) {
        double[] scores = new double[errors.length];

        // Find min and max
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (double error : errors) {
            if (error < min) min = error;
            if (error > max) max = error;
        }

        // Normalize to [0, 1]
        double range = max - min;
        if (range > 0) {
            for (int i = 0; i < errors.length; i++) {
                scores[i] = (errors[i] - min) / range;
            }
        }

        return scores;
    }

    /**
     * Classifies cells as normal (0) or anomaly (1) based on threshold.
     *
     * @param errors Array of reconstruction errors
     * @param threshold Threshold in standard deviations
     * @return Array of binary classifications
     */
    private int[] classifyAnomalies(double[] errors, double threshold) {
        int[] classifications = new int[errors.length];

        // Calculate mean and standard deviation
        double sum = 0.0;
        for (double error : errors) {
            sum += error;
        }
        double mean = sum / errors.length;

        double variance = 0.0;
        for (double error : errors) {
            double diff = error - mean;
            variance += diff * diff;
        }
        double stdDev = Math.sqrt(variance / errors.length);

        // Classify based on threshold
        double cutoff = mean + (threshold * stdDev);
        logger.info("Anomaly detection: mean={}, stdDev={}, cutoff={}", mean, stdDev, cutoff);

        int anomalyCount = 0;
        for (int i = 0; i < errors.length; i++) {
            if (errors[i] > cutoff) {
                classifications[i] = 1; // Anomaly
                anomalyCount++;
            } else {
                classifications[i] = 0; // Normal
            }
        }

        logger.info("Detected {} anomalies out of {} cells ({:.2f}%)",
                   anomalyCount, errors.length, (anomalyCount * 100.0 / errors.length));

        return classifications;
    }

    /**
     * Processes the anomaly detection using trained VAE model.
     *
     * @param al ArrayList containing GUI components with parameters
     * @param feature 2D array of existing features [objects][features]
     * @param validate whether to perform validation
     * @return true if successful, false otherwise
     */
    @Override
    public boolean process(ArrayList al, double[][] feature, boolean validate) {
        logger.info("Starting VAE anomaly detection...");

        try {
            // Extract parameters from GUI components
            JTextField modelPathField = (JTextField) al.get(1);
            modelPath = modelPathField.getText();

            JComboBox<?> modeCombo = (JComboBox<?>) al.get(4);
            outputMode = (String) modeCombo.getSelectedItem();

            JSpinner thresholdSpinner = (JSpinner) al.get(6);
            threshold = (double) thresholdSpinner.getValue();

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
            logger.info("Model input size: {}^3", inputSize);

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

            logger.info("Computing anomaly scores for {} objects using mode: {}",
                       objects.size(), outputMode);

            // Compute reconstruction errors for all objects
            double[] reconstructionErrors = new double[objects.size()];

            for (int i = 0; i < objects.size(); i++) {
                MicroObject cell = objects.get(i);

                // Extract 3D region around cell
                ImageStack region = extractor.extractRegion(cell, imageStack);

                // Convert to tensor
                Tensor inputTensor = converter.imageStackToTensor(region);

                // Encode and decode (full reconstruction)
                Tensor reconstruction = model.reconstruct(inputTensor);

                // Compute reconstruction error (MSE)
                double error = computeMSE(reconstruction, inputTensor);
                reconstructionErrors[i] = error;

                // Cleanup
                inputTensor.close();
                reconstruction.close();

                // Progress tracking
                if (i % 100 == 0) {
                    progress = (int) ((i / (double) objects.size()) * 100);
                    logger.debug("Computed reconstruction error for {}/{} objects", i, objects.size());
                }
            }

            // Process results based on output mode
            dataResult = new ArrayList();

            if (outputMode.contains("Binary")) {
                // Binary classification
                int[] classifications = classifyAnomalies(reconstructionErrors, threshold);
                for (int classification : classifications) {
                    dataResult.add(classification);
                }
            } else if (outputMode.contains("Score")) {
                // Normalized anomaly scores
                double[] scores = normalizeToScores(reconstructionErrors);
                for (double score : scores) {
                    dataResult.add(score);
                }
            } else {
                // Raw reconstruction errors
                for (double error : reconstructionErrors) {
                    dataResult.add(error);
                }
            }

            // Log statistics
            double[] sortedErrors = Arrays.copyOf(reconstructionErrors, reconstructionErrors.length);
            Arrays.sort(sortedErrors);
            double median = sortedErrors[sortedErrors.length / 2];
            double p95 = sortedErrors[(int) (sortedErrors.length * 0.95)];

            logger.info("Reconstruction error statistics: median={}, 95th percentile={}", median, p95);

            progress = 100;
            logger.info("VAE anomaly detection complete for {} objects", objects.size());

            return true;

        } catch (Exception e) {
            logger.error("Error during VAE anomaly detection", e);
            return false;
        } finally {
            // Cleanup resources
            if (model != null) {
                model.close();
            }
        }
    }
}
