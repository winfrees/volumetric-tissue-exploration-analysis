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
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JTextField;
import javax.swing.filechooser.FileNameExtensionFilter;
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
 * Feature extraction using 3D Variational Autoencoder latent space.
 * Encodes volumetric cell regions into low-dimensional latent features.
 *
 * @author VTEA Developer
 */
@Plugin(type = FeatureProcessing.class)
public class VAEFeatureExtraction extends AbstractFeatureProcessing {

    private static final Logger logger = LoggerFactory.getLogger(VAEFeatureExtraction.class);

    private VariationalAutoencoder3D model;
    private CellRegionExtractor extractor;
    private TensorConverter converter;
    private String modelPath;

    /**
     * Creates the Comment Text for the Block GUI.
     *
     * @param comComponents the parameters (Components) selected by the user
     * @return comment text detailing the parameters
     */
    public static String getBlockComment(ArrayList comComponents) {
        String comment = "<html>";
        comment = comment.concat("VAE Feature Extraction<br>");

        JTextField modelField = (JTextField) comComponents.get(1);
        String path = modelField.getText();
        if (path != null && !path.isEmpty()) {
            File f = new File(path);
            comment = comment.concat("Model: " + f.getName());
        } else {
            comment = comment.concat("Model: Not selected");
        }

        comment = comment.concat("</html>");
        return comment;
    }

    /**
     * Basic Constructor. Sets all protected variables.
     */
    public VAEFeatureExtraction() {
        VERSION = "1.0";
        AUTHOR = "VTEA Developer";
        COMMENT = "Extract latent features using trained 3D VAE";
        NAME = "VAE Feature Extraction";
        KEY = "VAEFeatures";
        TYPE = "Feature";
    }

    /**
     * Constructor called for initialization of Setup GUI.
     *
     * @param max the number of objects segmented in the volume (unused)
     */
    public VAEFeatureExtraction(int max) {
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
    }

    @Override
    public String getDataDescription(ArrayList params) {
        return "VAE_Latent";
    }

    /**
     * Processes the feature extraction using trained VAE model.
     * Extracts latent features for each cell in the dataset.
     *
     * @param al ArrayList containing GUI components with parameters
     * @param feature 2D array of existing features [objects][features]
     * @param validate whether to perform validation
     * @return true if successful, false otherwise
     */
    @Override
    public boolean process(ArrayList al, double[][] feature, boolean validate) {
        logger.info("Starting VAE feature extraction...");

        try {
            // Extract parameters from GUI components
            JTextField modelPathField = (JTextField) al.get(1);
            modelPath = modelPathField.getText();

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

            logger.info("Processing {} objects", objects.size());

            // Initialize result storage
            dataResult = new ArrayList();
            int latentDim = model.getConfig().getLatentDim();

            // Process each object
            for (int i = 0; i < objects.size(); i++) {
                MicroObject cell = objects.get(i);

                // Extract 3D region around cell
                ImageStack region = extractor.extractRegion(cell, imageStack);

                // Convert to tensor
                Tensor inputTensor = converter.imageStackToTensor(region);

                // Encode to latent space (using mean, not sampling)
                Tensor latentFeatures = model.encode(inputTensor);

                // Extract latent values as double array
                double[] latentArray = new double[latentDim];
                for (int j = 0; j < latentDim; j++) {
                    latentArray[j] = latentFeatures.index_get(j).item().doubleValue();
                }

                // Store as ArrayList for compatibility with VTEA
                ArrayList<Double> cellFeatures = new ArrayList<>();
                for (double val : latentArray) {
                    cellFeatures.add(val);
                }
                dataResult.add(cellFeatures);

                // Cleanup tensors
                inputTensor.close();
                latentFeatures.close();

                // Progress tracking
                if (i % 100 == 0) {
                    progress = (int) ((i / (double) objects.size()) * 100);
                    logger.debug("Processed {}/{} objects ({}%)", i, objects.size(), progress);
                }
            }

            progress = 100;
            logger.info("VAE feature extraction complete. Extracted {} features for {} objects",
                       latentDim, objects.size());

            return true;

        } catch (Exception e) {
            logger.error("Error during VAE feature extraction", e);
            return false;
        } finally {
            // Cleanup resources
            if (model != null) {
                model.close();
            }
        }
    }
}
