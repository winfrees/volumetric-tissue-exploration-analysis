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
package vtea.deeplearning.ui;

import vtea.deeplearning.models.Generic3DCNN;
import vtea.deeplearning.models.NephNet3D;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Panel for configuring deep learning model architecture and parameters.
 *
 * @author VTEA Deep Learning Team
 */
public class ModelConfigurationPanel extends JPanel {

    // Model selection
    private JComboBox<String> modelTypeCombo;
    private JTextField modelNameField;

    // Input configuration
    private JSpinner inputChannelsSpinner;
    private JSpinner numClassesSpinner;
    private JSpinner regionSizeSpinner;

    // Architecture configuration (for Generic3DCNN)
    private JTextField blockChannelsField;
    private JTextField kernelSizesField;
    private JTextField fcLayersField;
    private JComboBox<String> activationCombo;
    private JCheckBox batchNormCheckbox;
    private JSpinner dropoutSpinner;

    // Model info display
    private JTextArea modelInfoArea;

    /**
     * Constructor
     */
    public ModelConfigurationPanel() {
        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Title
        JLabel titleLabel = new JLabel("Model Configuration");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 16));
        add(titleLabel, BorderLayout.NORTH);

        // Main content panel
        JPanel contentPanel = new JPanel();
        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        // Add sections
        contentPanel.add(createBasicConfigSection());
        contentPanel.add(Box.createVerticalStrut(10));
        contentPanel.add(createArchitectureSection());
        contentPanel.add(Box.createVerticalStrut(10));
        contentPanel.add(createModelInfoSection());

        // Wrap in scroll pane
        JScrollPane scrollPane = new JScrollPane(contentPanel);
        scrollPane.setBorder(null);
        add(scrollPane, BorderLayout.CENTER);

        // Update button
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton updateButton = new JButton("Update Model Info");
        updateButton.addActionListener(e -> updateModelInfo());
        buttonPanel.add(updateButton);
        add(buttonPanel, BorderLayout.SOUTH);

        // Initialize
        updateModelInfo();
    }

    /**
     * Create basic configuration section
     */
    private JPanel createBasicConfigSection() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(new TitledBorder("Basic Configuration"));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;

        int row = 0;

        // Model type
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Model Type:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        modelTypeCombo = new JComboBox<>(new String[]{
            "NephNet3D",
            "Generic3DCNN - Lightweight",
            "Generic3DCNN - Deep",
            "Generic3DCNN - Wide",
            "Generic3DCNN - Custom"
        });
        modelTypeCombo.addActionListener(e -> onModelTypeChanged());
        panel.add(modelTypeCombo, gbc);

        row++;

        // Model name
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0.0;
        panel.add(new JLabel("Model Name:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        modelNameField = new JTextField("MyModel", 20);
        panel.add(modelNameField, gbc);

        row++;

        // Input channels
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0.0;
        panel.add(new JLabel("Input Channels:"), gbc);

        gbc.gridx = 1;
        inputChannelsSpinner = new JSpinner(new SpinnerNumberModel(2, 1, 10, 1));
        panel.add(inputChannelsSpinner, gbc);

        row++;

        // Number of classes
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Number of Classes:"), gbc);

        gbc.gridx = 1;
        numClassesSpinner = new JSpinner(new SpinnerNumberModel(3, 2, 20, 1));
        panel.add(numClassesSpinner, gbc);

        row++;

        // Region size
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Region Size (cubic):"), gbc);

        gbc.gridx = 1;
        regionSizeSpinner = new JSpinner(new SpinnerNumberModel(64, 16, 128, 8));
        panel.add(regionSizeSpinner, gbc);

        return panel;
    }

    /**
     * Create architecture configuration section
     */
    private JPanel createArchitectureSection() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(new TitledBorder("Architecture (Generic3DCNN)"));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;

        int row = 0;

        // Block channels
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Block Channels:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        blockChannelsField = new JTextField("32,64,128,256", 20);
        blockChannelsField.setToolTipText("Comma-separated channel counts for each conv block");
        panel.add(blockChannelsField, gbc);

        row++;

        // Kernel sizes
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0.0;
        panel.add(new JLabel("Kernel Sizes:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        kernelSizesField = new JTextField("3,3,3,3", 20);
        kernelSizesField.setToolTipText("Comma-separated kernel sizes for each block");
        panel.add(kernelSizesField, gbc);

        row++;

        // FC layers
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0.0;
        panel.add(new JLabel("FC Layers:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        fcLayersField = new JTextField("256,128", 20);
        fcLayersField.setToolTipText("Comma-separated hidden layer sizes");
        panel.add(fcLayersField, gbc);

        row++;

        // Activation
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0.0;
        panel.add(new JLabel("Activation:"), gbc);

        gbc.gridx = 1;
        activationCombo = new JComboBox<>(new String[]{"RELU", "LEAKY_RELU", "ELU"});
        activationCombo.setSelectedItem("LEAKY_RELU");
        panel.add(activationCombo, gbc);

        row++;

        // Batch normalization
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Batch Normalization:"), gbc);

        gbc.gridx = 1;
        batchNormCheckbox = new JCheckBox("", true);
        panel.add(batchNormCheckbox, gbc);

        row++;

        // Dropout rate
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Dropout Rate:"), gbc);

        gbc.gridx = 1;
        dropoutSpinner = new JSpinner(new SpinnerNumberModel(0.5, 0.0, 0.9, 0.1));
        panel.add(dropoutSpinner, gbc);

        return panel;
    }

    /**
     * Create model info section
     */
    private JPanel createModelInfoSection() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(new TitledBorder("Model Information"));

        modelInfoArea = new JTextArea(8, 40);
        modelInfoArea.setEditable(false);
        modelInfoArea.setFont(new Font("Monospaced", Font.PLAIN, 11));
        modelInfoArea.setBackground(new Color(245, 245, 245));

        JScrollPane scrollPane = new JScrollPane(modelInfoArea);
        panel.add(scrollPane, BorderLayout.CENTER);

        return panel;
    }

    /**
     * Handle model type change
     */
    private void onModelTypeChanged() {
        String selected = (String) modelTypeCombo.getSelectedItem();
        boolean isCustom = selected.equals("Generic3DCNN - Custom");

        // Enable/disable custom architecture fields
        blockChannelsField.setEnabled(isCustom);
        kernelSizesField.setEnabled(isCustom);
        fcLayersField.setEnabled(isCustom);
        activationCombo.setEnabled(isCustom);
        batchNormCheckbox.setEnabled(isCustom);
        dropoutSpinner.setEnabled(isCustom);

        // Set defaults for predefined architectures
        if (!isCustom) {
            switch (selected) {
                case "NephNet3D":
                    blockChannelsField.setText("32,64,128,256");
                    kernelSizesField.setText("3,3,3,3");
                    fcLayersField.setText("256,128");
                    activationCombo.setSelectedItem("LEAKY_RELU");
                    batchNormCheckbox.setSelected(true);
                    dropoutSpinner.setValue(0.5);
                    break;
                case "Generic3DCNN - Lightweight":
                    blockChannelsField.setText("16,32,64");
                    kernelSizesField.setText("3,3,3");
                    fcLayersField.setText("128");
                    activationCombo.setSelectedItem("RELU");
                    batchNormCheckbox.setSelected(true);
                    dropoutSpinner.setValue(0.3);
                    break;
                case "Generic3DCNN - Deep":
                    blockChannelsField.setText("32,64,128,256,512");
                    kernelSizesField.setText("3,3,3,3,3");
                    fcLayersField.setText("512,256,128");
                    activationCombo.setSelectedItem("RELU");
                    batchNormCheckbox.setSelected(true);
                    dropoutSpinner.setValue(0.5);
                    break;
                case "Generic3DCNN - Wide":
                    blockChannelsField.setText("64,128,256,512");
                    kernelSizesField.setText("3,3,3,3");
                    fcLayersField.setText("512,256");
                    activationCombo.setSelectedItem("RELU");
                    batchNormCheckbox.setSelected(true);
                    dropoutSpinner.setValue(0.5);
                    break;
            }
        }

        updateModelInfo();
    }

    /**
     * Update model information display
     */
    private void updateModelInfo() {
        try {
            StringBuilder info = new StringBuilder();
            String modelType = (String) modelTypeCombo.getSelectedItem();

            info.append("Model Type: ").append(modelType).append("\n");
            info.append("Model Name: ").append(modelNameField.getText()).append("\n\n");

            info.append("Input Configuration:\n");
            info.append("  Channels: ").append(inputChannelsSpinner.getValue()).append("\n");
            info.append("  Classes: ").append(numClassesSpinner.getValue()).append("\n");
            info.append("  Region Size: ").append(regionSizeSpinner.getValue()).append("³\n\n");

            if (modelType.startsWith("Generic3DCNN")) {
                info.append("Architecture:\n");
                info.append("  Conv Blocks: ").append(blockChannelsField.getText()).append("\n");
                info.append("  Kernel Sizes: ").append(kernelSizesField.getText()).append("\n");
                info.append("  FC Layers: ").append(fcLayersField.getText()).append("\n");
                info.append("  Activation: ").append(activationCombo.getSelectedItem()).append("\n");
                info.append("  Batch Norm: ").append(batchNormCheckbox.isSelected() ? "Yes" : "No").append("\n");
                info.append("  Dropout: ").append(dropoutSpinner.getValue()).append("\n\n");

                // Estimate parameters
                int[] channels = parseIntArray(blockChannelsField.getText());
                info.append("Estimated Parameters: ").append(estimateParameters(channels)).append("\n");
                info.append("Estimated Memory: ").append(String.format("%.1f MB", estimateMemoryMB(channels)));
            } else if (modelType.equals("NephNet3D")) {
                info.append("Architecture: NephNet3D Standard\n");
                info.append("  Conv Blocks: 32 → 64 → 128 → 256\n");
                info.append("  Kernel Size: 3×3×3\n");
                info.append("  FC Layers: 256 → 128\n");
                info.append("  Activation: LeakyReLU\n");
                info.append("  Batch Norm: Yes\n");
                info.append("  Dropout: 0.5\n\n");
                info.append("Estimated Parameters: ~2.1M\n");
                info.append("Estimated Memory: ~8.4 MB");
            }

            modelInfoArea.setText(info.toString());

        } catch (Exception e) {
            modelInfoArea.setText("Error: " + e.getMessage());
        }
    }

    /**
     * Parse comma-separated integers
     */
    private int[] parseIntArray(String text) {
        String[] parts = text.split(",");
        int[] result = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Integer.parseInt(parts[i].trim());
        }
        return result;
    }

    /**
     * Estimate parameter count (rough approximation)
     */
    private String estimateParameters(int[] channels) {
        long params = 0;
        int inputChannels = (Integer) inputChannelsSpinner.getValue();

        // Conv layers
        params += inputChannels * channels[0] * 27;  // First conv
        for (int i = 1; i < channels.length; i++) {
            params += channels[i-1] * channels[i] * 27;  // 3x3x3 kernel
        }

        // FC layers
        int[] fcLayers = parseIntArray(fcLayersField.getText());
        int fcInput = channels[channels.length - 1];
        for (int fcSize : fcLayers) {
            params += fcInput * fcSize;
            fcInput = fcSize;
        }
        params += fcInput * (Integer) numClassesSpinner.getValue();

        if (params >= 1_000_000) {
            return String.format("%.1fM", params / 1_000_000.0);
        } else if (params >= 1_000) {
            return String.format("%.1fK", params / 1_000.0);
        } else {
            return String.valueOf(params);
        }
    }

    /**
     * Estimate memory usage
     */
    private double estimateMemoryMB(int[] channels) {
        long params = 0;
        int inputChannels = (Integer) inputChannelsSpinner.getValue();

        params += inputChannels * channels[0] * 27;
        for (int i = 1; i < channels.length; i++) {
            params += channels[i-1] * channels[i] * 27;
        }

        int[] fcLayers = parseIntArray(fcLayersField.getText());
        int fcInput = channels[channels.length - 1];
        for (int fcSize : fcLayers) {
            params += fcInput * fcSize;
            fcInput = fcSize;
        }
        params += fcInput * (Integer) numClassesSpinner.getValue();

        return (params * 4.0) / (1024.0 * 1024.0);  // 4 bytes per float32
    }

    /**
     * Get configuration as map
     */
    public Map<String, Object> getConfiguration() {
        Map<String, Object> config = new HashMap<>();

        config.put("modelType", modelTypeCombo.getSelectedItem());
        config.put("modelName", modelNameField.getText());
        config.put("inputChannels", inputChannelsSpinner.getValue());
        config.put("numClasses", numClassesSpinner.getValue());
        config.put("regionSize", regionSizeSpinner.getValue());

        config.put("blockChannels", blockChannelsField.getText());
        config.put("kernelSizes", kernelSizesField.getText());
        config.put("fcLayers", fcLayersField.getText());
        config.put("activation", activationCombo.getSelectedItem());
        config.put("batchNorm", batchNormCheckbox.isSelected());
        config.put("dropout", dropoutSpinner.getValue());

        return config;
    }

    /**
     * Test main
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Model Configuration");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new ModelConfigurationPanel());
            frame.setSize(600, 700);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
