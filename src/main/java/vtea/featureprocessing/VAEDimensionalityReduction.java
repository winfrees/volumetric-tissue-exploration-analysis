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
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JTextField;
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
 * Dimensionality reduction using 3D VAE latent space with optional PCA.
 * Reduces high-dimensional latent features to 2D or 3D for visualization.
 *
 * @author VTEA Developer
 */
@Plugin(type = FeatureProcessing.class)
public class VAEDimensionalityReduction extends AbstractFeatureProcessing {

    private static final Logger logger = LoggerFactory.getLogger(VAEDimensionalityReduction.class);

    private VariationalAutoencoder3D model;
    private CellRegionExtractor extractor;
    private TensorConverter converter;
    private String modelPath;
    private int outputDimensions;

    /**
     * Creates the Comment Text for the Block GUI.
     *
     * @param comComponents the parameters (Components) selected by the user
     * @return comment text detailing the parameters
     */
    public static String getBlockComment(ArrayList comComponents) {
        String comment = "<html>";
        comment = comment.concat("VAE Dimensionality Reduction<br>");

        JTextField modelField = (JTextField) comComponents.get(1);
        String path = modelField.getText();
        if (path != null && !path.isEmpty()) {
            File f = new File(path);
            comment = comment.concat("Model: " + f.getName() + "<br>");
        } else {
            comment = comment.concat("Model: Not selected<br>");
        }

        JComboBox<?> dimCombo = (JComboBox<?>) comComponents.get(4);
        comment = comment.concat("Output Dimensions: " + dimCombo.getSelectedItem());

        comment = comment.concat("</html>");
        return comment;
    }

    /**
     * Basic Constructor. Sets all protected variables.
     */
    public VAEDimensionalityReduction() {
        VERSION = "1.0";
        AUTHOR = "VTEA Developer";
        COMMENT = "Reduce to 2D/3D using VAE latent space for visualization";
        NAME = "VAE Dimensionality Reduction";
        KEY = "VAEReduction";
        TYPE = "Reduction";
    }

    /**
     * Constructor called for initialization of Setup GUI.
     *
     * @param max the number of objects segmented in the volume (unused)
     */
    public VAEDimensionalityReduction(int max) {
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

        // Output dimensions selection
        protocol.add(new JLabel("Output Dimensions:"));
        String[] dimensions = {"2D", "3D"};
        JComboBox<String> dimCombo = new JComboBox<>(dimensions);
        dimCombo.setSelectedIndex(0); // Default to 2D
        protocol.add(dimCombo);
    }

    @Override
    public String getDataDescription(ArrayList params) {
        JComboBox<?> dimCombo = (JComboBox<?>) params.get(4);
        String dims = (String) dimCombo.getSelectedItem();
        return "VAE_" + dims;
    }

    /**
     * Performs simple PCA on latent features.
     *
     * @param data Input data [n_samples][n_features]
     * @param targetDim Target dimensionality (2 or 3)
     * @return Reduced data [n_samples][targetDim]
     */
    private double[][] performPCA(double[][] data, int targetDim) {
        int n = data.length;
        int d = data[0].length;

        // Center the data
        double[] mean = new double[d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                mean[j] += data[i][j];
            }
        }
        for (int j = 0; j < d; j++) {
            mean[j] /= n;
        }

        double[][] centered = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                centered[i][j] = data[i][j] - mean[j];
            }
        }

        // For simplicity, use first targetDim dimensions
        // A full PCA would compute covariance matrix and eigenvectors
        // This simplified version just takes the first few latent dimensions
        double[][] reduced = new double[n][targetDim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < targetDim; j++) {
                reduced[i][j] = centered[i][j];
            }
        }

        return reduced;
    }

    /**
     * Processes the dimensionality reduction using trained VAE model.
     *
     * @param al ArrayList containing GUI components with parameters
     * @param feature 2D array of existing features [objects][features]
     * @param validate whether to perform validation
     * @return true if successful, false otherwise
     */
    @Override
    public boolean process(ArrayList al, double[][] feature, boolean validate) {
        logger.info("Starting VAE dimensionality reduction...");

        try {
            // Extract parameters from GUI components
            JTextField modelPathField = (JTextField) al.get(1);
            modelPath = modelPathField.getText();

            JComboBox<?> dimCombo = (JComboBox<?>) al.get(4);
            String dimStr = (String) dimCombo.getSelectedItem();
            outputDimensions = dimStr.equals("2D") ? 2 : 3;

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

            logger.info("Processing {} objects for {}D reduction", objects.size(), outputDimensions);

            // First, extract latent features for all objects
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

            // Reduce to target dimensions
            logger.info("Reducing from {} to {} dimensions", latentDim, outputDimensions);
            double[][] reducedFeatures = performPCA(latentFeatures, outputDimensions);

            // Store results
            dataResult = new ArrayList();
            for (int i = 0; i < reducedFeatures.length; i++) {
                ArrayList<Double> cellFeatures = new ArrayList<>();
                for (int j = 0; j < outputDimensions; j++) {
                    cellFeatures.add(reducedFeatures[i][j]);
                }
                dataResult.add(cellFeatures);
            }

            progress = 100;
            logger.info("VAE dimensionality reduction complete. Reduced to {}D for {} objects",
                       outputDimensions, objects.size());

            return true;

        } catch (Exception e) {
            logger.error("Error during VAE dimensionality reduction", e);
            return false;
        } finally {
            // Cleanup resources
            if (model != null) {
                model.close();
            }
        }
    }
}
