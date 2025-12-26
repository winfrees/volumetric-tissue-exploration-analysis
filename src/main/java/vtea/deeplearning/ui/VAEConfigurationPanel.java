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
package vtea.deeplearning.ui;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.io.File;
import vtea.deeplearning.models.VAEConfig;

/**
 * Configuration panel for VAE model settings.
 * Provides GUI controls for all VAE hyperparameters.
 *
 * @author VTEA Developer
 */
public class VAEConfigurationPanel extends JPanel {

    // Architecture settings
    private JComboBox<String> architectureCombo;
    private JSpinner inputSizeSpinner;
    private JSpinner latentDimSpinner;
    private JTextField encoderChannelsField;
    private JTextField decoderChannelsField;

    // Training settings
    private JSpinner batchSizeSpinner;
    private JSpinner epochsSpinner;
    private JSpinner learningRateSpinner;
    private JCheckBox useGPUCheckbox;

    // VAE-specific settings
    private JSpinner betaSpinner;
    private JSpinner klWarmupEpochsSpinner;
    private JComboBox<String> reconstructionLossCombo;

    // Output settings
    private JTextField checkpointDirField;
    private JButton browseDirButton;

    /**
     * Creates a new VAE configuration panel.
     */
    public VAEConfigurationPanel() {
        initComponents();
    }

    private void initComponents() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Architecture panel
        add(createArchitecturePanel());
        add(Box.createVerticalStrut(10));

        // Training parameters panel
        add(createTrainingPanel());
        add(Box.createVerticalStrut(10));

        // VAE-specific panel
        add(createVAEPanel());
        add(Box.createVerticalStrut(10));

        // Output panel
        add(createOutputPanel());
    }

    private JPanel createArchitecturePanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "Model Architecture",
            TitledBorder.LEFT, TitledBorder.TOP));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        // Architecture preset
        gbc.gridx = 0; gbc.gridy = 0;
        panel.add(new JLabel("Architecture:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        String[] architectures = {"SMALL (32³, 16D)", "MEDIUM (64³, 32D)", "LARGE (128³, 64D)", "CUSTOM"};
        architectureCombo = new JComboBox<>(architectures);
        architectureCombo.setSelectedIndex(1); // Default to MEDIUM
        architectureCombo.addActionListener(e -> onArchitectureChanged());
        panel.add(architectureCombo, gbc);

        // Input size
        gbc.gridx = 0; gbc.gridy = 1; gbc.weightx = 0;
        panel.add(new JLabel("Input Size (voxels):"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        inputSizeSpinner = new JSpinner(new SpinnerNumberModel(64, 16, 256, 8));
        inputSizeSpinner.setEnabled(false);
        panel.add(inputSizeSpinner, gbc);

        // Latent dimension
        gbc.gridx = 0; gbc.gridy = 2; gbc.weightx = 0;
        panel.add(new JLabel("Latent Dimension:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        latentDimSpinner = new JSpinner(new SpinnerNumberModel(32, 4, 256, 4));
        latentDimSpinner.setEnabled(false);
        panel.add(latentDimSpinner, gbc);

        // Encoder channels
        gbc.gridx = 0; gbc.gridy = 3; gbc.weightx = 0;
        panel.add(new JLabel("Encoder Channels:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        encoderChannelsField = new JTextField("32,64,128,256");
        encoderChannelsField.setEnabled(false);
        panel.add(encoderChannelsField, gbc);

        // Decoder channels
        gbc.gridx = 0; gbc.gridy = 4; gbc.weightx = 0;
        panel.add(new JLabel("Decoder Channels:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        decoderChannelsField = new JTextField("256,128,64,32");
        decoderChannelsField.setEnabled(false);
        panel.add(decoderChannelsField, gbc);

        return panel;
    }

    private JPanel createTrainingPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "Training Parameters",
            TitledBorder.LEFT, TitledBorder.TOP));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        // Batch size
        gbc.gridx = 0; gbc.gridy = 0;
        panel.add(new JLabel("Batch Size:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        batchSizeSpinner = new JSpinner(new SpinnerNumberModel(8, 1, 128, 1));
        panel.add(batchSizeSpinner, gbc);

        // Epochs
        gbc.gridx = 0; gbc.gridy = 1; gbc.weightx = 0;
        panel.add(new JLabel("Epochs:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        epochsSpinner = new JSpinner(new SpinnerNumberModel(100, 1, 1000, 10));
        panel.add(epochsSpinner, gbc);

        // Learning rate
        gbc.gridx = 0; gbc.gridy = 2; gbc.weightx = 0;
        panel.add(new JLabel("Learning Rate:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        learningRateSpinner = new JSpinner(new SpinnerNumberModel(0.001, 0.00001, 0.1, 0.0001));
        JSpinner.NumberEditor editor = new JSpinner.NumberEditor(learningRateSpinner, "0.#####");
        learningRateSpinner.setEditor(editor);
        panel.add(learningRateSpinner, gbc);

        // Use GPU
        gbc.gridx = 0; gbc.gridy = 3; gbc.weightx = 0;
        panel.add(new JLabel("Use GPU:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        useGPUCheckbox = new JCheckBox();
        useGPUCheckbox.setSelected(true);
        panel.add(useGPUCheckbox, gbc);

        return panel;
    }

    private JPanel createVAEPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "VAE Settings",
            TitledBorder.LEFT, TitledBorder.TOP));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        // Beta (KL weight)
        gbc.gridx = 0; gbc.gridy = 0;
        panel.add(new JLabel("Beta (KL weight):"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        betaSpinner = new JSpinner(new SpinnerNumberModel(1.0, 0.1, 10.0, 0.1));
        JSpinner.NumberEditor betaEditor = new JSpinner.NumberEditor(betaSpinner, "0.#");
        betaSpinner.setEditor(betaEditor);
        panel.add(betaSpinner, gbc);

        // KL warmup epochs
        gbc.gridx = 0; gbc.gridy = 1; gbc.weightx = 0;
        panel.add(new JLabel("KL Warmup Epochs:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        klWarmupEpochsSpinner = new JSpinner(new SpinnerNumberModel(10, 0, 100, 5));
        panel.add(klWarmupEpochsSpinner, gbc);

        // Reconstruction loss type
        gbc.gridx = 0; gbc.gridy = 2; gbc.weightx = 0;
        panel.add(new JLabel("Reconstruction Loss:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        String[] lossTypes = {"MSE", "BCE", "L1"};
        reconstructionLossCombo = new JComboBox<>(lossTypes);
        reconstructionLossCombo.setSelectedIndex(0);
        panel.add(reconstructionLossCombo, gbc);

        return panel;
    }

    private JPanel createOutputPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "Output Settings",
            TitledBorder.LEFT, TitledBorder.TOP));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        // Checkpoint directory
        gbc.gridx = 0; gbc.gridy = 0;
        panel.add(new JLabel("Checkpoint Dir:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        checkpointDirField = new JTextField(System.getProperty("user.home") + "/vae_checkpoints");
        panel.add(checkpointDirField, gbc);

        gbc.gridx = 2; gbc.weightx = 0;
        browseDirButton = new JButton("Browse...");
        browseDirButton.addActionListener(e -> browseCheckpointDir());
        panel.add(browseDirButton, gbc);

        return panel;
    }

    private void onArchitectureChanged() {
        String selected = (String) architectureCombo.getSelectedItem();
        boolean isCustom = selected.contains("CUSTOM");

        inputSizeSpinner.setEnabled(isCustom);
        latentDimSpinner.setEnabled(isCustom);
        encoderChannelsField.setEnabled(isCustom);
        decoderChannelsField.setEnabled(isCustom);

        if (!isCustom) {
            if (selected.contains("SMALL")) {
                inputSizeSpinner.setValue(32);
                latentDimSpinner.setValue(16);
                encoderChannelsField.setText("16,32,64,128");
                decoderChannelsField.setText("128,64,32,16");
            } else if (selected.contains("MEDIUM")) {
                inputSizeSpinner.setValue(64);
                latentDimSpinner.setValue(32);
                encoderChannelsField.setText("32,64,128,256");
                decoderChannelsField.setText("256,128,64,32");
            } else if (selected.contains("LARGE")) {
                inputSizeSpinner.setValue(128);
                latentDimSpinner.setValue(64);
                encoderChannelsField.setText("64,128,256,512");
                decoderChannelsField.setText("512,256,128,64");
            }
        }
    }

    private void browseCheckpointDir() {
        JFileChooser chooser = new JFileChooser(checkpointDirField.getText());
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        chooser.setDialogTitle("Select Checkpoint Directory");

        int result = chooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            checkpointDirField.setText(chooser.getSelectedFile().getAbsolutePath());
        }
    }

    /**
     * Creates a VAEConfig from the current GUI settings.
     *
     * @return Configured VAEConfig object
     */
    public VAEConfig getConfig() {
        VAEConfig config = new VAEConfig();

        String archStr = (String) architectureCombo.getSelectedItem();
        if (archStr.contains("SMALL")) {
            config.setArchitecture(VAEConfig.VAEArchitecture.SMALL);
        } else if (archStr.contains("MEDIUM")) {
            config.setArchitecture(VAEConfig.VAEArchitecture.MEDIUM);
        } else if (archStr.contains("LARGE")) {
            config.setArchitecture(VAEConfig.VAEArchitecture.LARGE);
        } else {
            config.setArchitecture(VAEConfig.VAEArchitecture.CUSTOM);
            config.setInputSize((int) inputSizeSpinner.getValue());
            config.setLatentDim((int) latentDimSpinner.getValue());
            config.setEncoderChannels(parseChannels(encoderChannelsField.getText()));
            config.setDecoderChannels(parseChannels(decoderChannelsField.getText()));
        }

        config.setBatchSize((int) batchSizeSpinner.getValue());
        config.setNumEpochs((int) epochsSpinner.getValue());
        config.setLearningRate((double) learningRateSpinner.getValue());
        config.setUseGPU(useGPUCheckbox.isSelected());

        config.setBeta((double) betaSpinner.getValue());
        config.setKlWarmupEpochs((int) klWarmupEpochsSpinner.getValue());

        return config;
    }

    /**
     * Sets the configuration panel from a VAEConfig object.
     *
     * @param config Configuration to load
     */
    public void setConfig(VAEConfig config) {
        // Set architecture
        switch (config.getArchitecture()) {
            case SMALL:
                architectureCombo.setSelectedIndex(0);
                break;
            case MEDIUM:
                architectureCombo.setSelectedIndex(1);
                break;
            case LARGE:
                architectureCombo.setSelectedIndex(2);
                break;
            case CUSTOM:
                architectureCombo.setSelectedIndex(3);
                inputSizeSpinner.setValue(config.getInputSize());
                latentDimSpinner.setValue(config.getLatentDim());
                encoderChannelsField.setText(arrayToString(config.getEncoderChannels()));
                decoderChannelsField.setText(arrayToString(config.getDecoderChannels()));
                break;
        }

        batchSizeSpinner.setValue(config.getBatchSize());
        epochsSpinner.setValue(config.getNumEpochs());
        learningRateSpinner.setValue(config.getLearningRate());
        useGPUCheckbox.setSelected(config.isUseGPU());

        betaSpinner.setValue(config.getBeta());
        klWarmupEpochsSpinner.setValue(config.getKlWarmupEpochs());
    }

    public String getCheckpointDir() {
        return checkpointDirField.getText();
    }

    public void setCheckpointDir(String dir) {
        checkpointDirField.setText(dir);
    }

    public String getReconstructionLossType() {
        return (String) reconstructionLossCombo.getSelectedItem();
    }

    private int[] parseChannels(String text) {
        String[] parts = text.split(",");
        int[] channels = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            channels[i] = Integer.parseInt(parts[i].trim());
        }
        return channels;
    }

    private String arrayToString(int[] array) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(array[i]);
        }
        return sb.toString();
    }
}
