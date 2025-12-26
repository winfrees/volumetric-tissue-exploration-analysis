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
package vtea.deeplearning.plugins;

import vtea.deeplearning.DeepLearningConfig;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;

/**
 * Setup GUI for Deep Learning Classification plugin.
 * Provides interface for selecting model and configuring parameters.
 *
 * @author VTEA Deep Learning Team
 */
public class DeepLearningClassificationSetup extends JPanel implements ActionListener {

    private JTextField modelPathField;
    private JButton browseButton;
    private JCheckBox returnClassNamesCheckbox;
    private JSpinner batchSizeSpinner;
    private JLabel modelInfoLabel;
    private JLabel deviceInfoLabel;

    /**
     * Constructor
     */
    public DeepLearningClassificationSetup() {
        setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        // Title
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.gridwidth = 3;
        JLabel titleLabel = new JLabel("<html><b>Deep Learning Classification Setup</b></html>");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 14));
        add(titleLabel, gbc);

        // Model path label
        gbc.gridy = 1;
        gbc.gridwidth = 1;
        JLabel modelLabel = new JLabel("Model Path:");
        add(modelLabel, gbc);

        // Model path text field
        gbc.gridx = 1;
        gbc.weightx = 1.0;
        modelPathField = new JTextField(30);
        modelPathField.setToolTipText("Path to trained model file (without .pt extension)");
        add(modelPathField, gbc);

        // Browse button
        gbc.gridx = 2;
        gbc.weightx = 0.0;
        browseButton = new JButton("Browse...");
        browseButton.addActionListener(this);
        add(browseButton, gbc);

        // Model info label
        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.gridwidth = 3;
        modelInfoLabel = new JLabel(" ");
        modelInfoLabel.setFont(new Font("Arial", Font.ITALIC, 11));
        add(modelInfoLabel, gbc);

        // Return class names checkbox
        gbc.gridy = 3;
        returnClassNamesCheckbox = new JCheckBox("Return Class Names (instead of IDs)", true);
        returnClassNamesCheckbox.setToolTipText("If checked, returns human-readable class names; otherwise returns numeric class IDs");
        add(returnClassNamesCheckbox, gbc);

        // Batch size label
        gbc.gridy = 4;
        gbc.gridwidth = 1;
        JLabel batchLabel = new JLabel("Batch Size:");
        add(batchLabel, gbc);

        // Batch size spinner
        gbc.gridx = 1;
        gbc.gridwidth = 2;
        SpinnerNumberModel batchModel = new SpinnerNumberModel(16, 1, 128, 1);
        batchSizeSpinner = new JSpinner(batchModel);
        batchSizeSpinner.setToolTipText("Number of objects to process simultaneously (higher = faster but more memory)");
        add(batchSizeSpinner, gbc);

        // Device info
        gbc.gridx = 0;
        gbc.gridy = 5;
        gbc.gridwidth = 3;
        DeepLearningConfig config = DeepLearningConfig.getInstance();
        String device = config.isGpuAvailable() ? "GPU (CUDA)" : "CPU";
        deviceInfoLabel = new JLabel("Compute Device: " + device);
        deviceInfoLabel.setFont(new Font("Arial", Font.PLAIN, 11));
        add(deviceInfoLabel, gbc);

        // Help text
        gbc.gridy = 6;
        JTextArea helpText = new JTextArea(
            "This plugin uses trained 3D deep learning models to classify segmented cells.\n" +
            "Select a model trained using the VTEA Deep Learning Training interface.\n" +
            "The model will extract 3D regions around each cell and predict its class."
        );
        helpText.setEditable(false);
        helpText.setWrapStyleWord(true);
        helpText.setLineWrap(true);
        helpText.setOpaque(false);
        helpText.setFont(new Font("Arial", Font.PLAIN, 11));
        helpText.setRows(3);
        add(helpText, gbc);

        // Set default model path
        String defaultModelDir = config.getModelDirectory();
        File modelDir = new File(defaultModelDir);
        if (modelDir.exists() && modelDir.isDirectory()) {
            File[] models = modelDir.listFiles((dir, name) -> name.endsWith(".pt"));
            if (models != null && models.length > 0) {
                // Remove .pt extension
                String path = models[0].getAbsolutePath();
                path = path.substring(0, path.length() - 3);
                modelPathField.setText(path);
                updateModelInfo(path);
            }
        }
    }

    /**
     * Get setup components as ArrayList
     */
    public ArrayList<Component> getComponents() {
        ArrayList<Component> components = new ArrayList<>();

        // Add a label for normalization (placeholder)
        components.add(new JLabel("Normalize:"));

        // Model path field
        components.add(modelPathField);

        // Return class names checkbox
        components.add(returnClassNamesCheckbox);

        // Batch size spinner
        components.add(batchSizeSpinner);

        return components;
    }

    /**
     * Handle button actions
     */
    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == browseButton) {
            browseForModel();
        }
    }

    /**
     * Browse for model file
     */
    private void browseForModel() {
        DeepLearningConfig config = DeepLearningConfig.getInstance();
        String modelDir = config.getModelDirectory();

        JFileChooser fileChooser = new JFileChooser(modelDir);
        fileChooser.setDialogTitle("Select Trained Model");
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        // Filter for .pt files
        FileNameExtensionFilter filter = new FileNameExtensionFilter("PyTorch Model Files (*.pt)", "pt");
        fileChooser.setFileFilter(filter);

        int result = fileChooser.showOpenDialog(this);

        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            String path = selectedFile.getAbsolutePath();

            // Remove .pt extension if present
            if (path.endsWith(".pt")) {
                path = path.substring(0, path.length() - 3);
            }

            modelPathField.setText(path);
            updateModelInfo(path);
        }
    }

    /**
     * Update model info label
     */
    private void updateModelInfo(String modelPath) {
        File metaFile = new File(modelPath + ".meta");
        File ptFile = new File(modelPath + ".pt");

        if (ptFile.exists() && metaFile.exists()) {
            String modelName = new File(modelPath).getName();
            double sizeMB = ptFile.length() / (1024.0 * 1024.0);
            modelInfoLabel.setText(String.format("Model: %s (%.1f MB)", modelName, sizeMB));
            modelInfoLabel.setForeground(new Color(0, 128, 0));  // Green
        } else {
            modelInfoLabel.setText("⚠ Model files not found");
            modelInfoLabel.setForeground(Color.RED);
        }
    }

    /**
     * Create a test panel for standalone testing
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Deep Learning Classification Setup");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            DeepLearningClassificationSetup setup = new DeepLearningClassificationSetup();
            frame.add(setup);

            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
