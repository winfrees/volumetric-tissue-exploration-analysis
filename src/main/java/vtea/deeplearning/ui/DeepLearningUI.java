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

import vtea.deeplearning.DeepLearningConfig;

import javax.swing.*;
import java.awt.*;

/**
 * Integrated Deep Learning UI for VTEA.
 * Combines model configuration, training, and inference panels.
 *
 * @author VTEA Deep Learning Team
 */
public class DeepLearningUI extends JFrame {

    private ModelConfigurationPanel modelConfigPanel;
    private TrainingPanel trainingPanel;
    private JTabbedPane mainTabbedPane;

    /**
     * Constructor
     */
    public DeepLearningUI() {
        super("VTEA Deep Learning");

        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setLayout(new BorderLayout());

        // Create menu bar
        setJMenuBar(createMenuBar());

        // Create main tabbed pane
        mainTabbedPane = new JTabbedPane();

        // Add tabs
        modelConfigPanel = new ModelConfigurationPanel();
        mainTabbedPane.addTab("Model Configuration", modelConfigPanel);

        trainingPanel = new TrainingPanel();
        mainTabbedPane.addTab("Training", trainingPanel);

        add(mainTabbedPane, BorderLayout.CENTER);

        // Create status bar
        add(createStatusBar(), BorderLayout.SOUTH);

        // Set size and show
        setSize(1000, 800);
        setLocationRelativeTo(null);
    }

    /**
     * Create menu bar
     */
    private JMenuBar createMenuBar() {
        JMenuBar menuBar = new JMenuBar();

        // File menu
        JMenu fileMenu = new JMenu("File");

        JMenuItem loadMenuItem = new JMenuItem("Load Model...");
        loadMenuItem.addActionListener(e -> onLoadModel());
        fileMenu.add(loadMenuItem);

        JMenuItem saveMenuItem = new JMenuItem("Save Model...");
        saveMenuItem.addActionListener(e -> onSaveModel());
        fileMenu.add(saveMenuItem);

        fileMenu.addSeparator();

        JMenuItem exitMenuItem = new JMenuItem("Exit");
        exitMenuItem.addActionListener(e -> dispose());
        fileMenu.add(exitMenuItem);

        menuBar.add(fileMenu);

        // Tools menu
        JMenu toolsMenu = new JMenu("Tools");

        JMenuItem settingsMenuItem = new JMenuItem("Settings...");
        settingsMenuItem.addActionListener(e -> onShowSettings());
        toolsMenu.add(settingsMenuItem);

        menuBar.add(toolsMenu);

        // Help menu
        JMenu helpMenu = new JMenu("Help");

        JMenuItem aboutMenuItem = new JMenuItem("About");
        aboutMenuItem.addActionListener(e -> onShowAbout());
        helpMenu.add(aboutMenuItem);

        JMenuItem docsMenuItem = new JMenuItem("Documentation");
        docsMenuItem.addActionListener(e -> onShowDocumentation());
        helpMenu.add(docsMenuItem);

        menuBar.add(helpMenu);

        return menuBar;
    }

    /**
     * Create status bar
     */
    private JPanel createStatusBar() {
        JPanel statusBar = new JPanel(new BorderLayout());
        statusBar.setBorder(BorderFactory.createEtchedBorder());

        DeepLearningConfig config = DeepLearningConfig.getInstance();

        JLabel deviceLabel = new JLabel(" Device: " + (config.isGpuAvailable() ? "GPU (CUDA)" : "CPU"));
        statusBar.add(deviceLabel, BorderLayout.WEST);

        JLabel modelDirLabel = new JLabel("Model Directory: " + config.getModelDirectory() + " ");
        statusBar.add(modelDirLabel, BorderLayout.EAST);

        return statusBar;
    }

    /**
     * Menu actions
     */
    private void onLoadModel() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Load Model");

        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            // Load model logic here
            JOptionPane.showMessageDialog(this,
                "Model loading functionality to be implemented",
                "Load Model",
                JOptionPane.INFORMATION_MESSAGE);
        }
    }

    private void onSaveModel() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Model");

        int result = fileChooser.showSaveDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            // Save model logic here
            JOptionPane.showMessageDialog(this,
                "Model saving functionality to be implemented",
                "Save Model",
                JOptionPane.INFORMATION_MESSAGE);
        }
    }

    private void onShowSettings() {
        JDialog settingsDialog = new JDialog(this, "Settings", true);
        settingsDialog.setLayout(new BorderLayout(10, 10));

        JPanel panel = new JPanel(new GridLayout(3, 2, 10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        DeepLearningConfig config = DeepLearningConfig.getInstance();

        panel.add(new JLabel("Model Directory:"));
        JTextField modelDirField = new JTextField(config.getModelDirectory());
        panel.add(modelDirField);

        panel.add(new JLabel("Default Batch Size:"));
        JSpinner batchSpinner = new JSpinner(new SpinnerNumberModel(config.getDefaultBatchSize(), 1, 128, 1));
        panel.add(batchSpinner);

        panel.add(new JLabel("Default Region Size:"));
        JSpinner regionSpinner = new JSpinner(new SpinnerNumberModel(config.getDefaultRegionSize(), 16, 128, 8));
        panel.add(regionSpinner);

        settingsDialog.add(panel, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton okButton = new JButton("OK");
        okButton.addActionListener(e -> {
            config.setModelDirectory(modelDirField.getText());
            config.setDefaultBatchSize((Integer) batchSpinner.getValue());
            config.setDefaultRegionSize((Integer) regionSpinner.getValue());
            settingsDialog.dispose();
        });
        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(e -> settingsDialog.dispose());

        buttonPanel.add(okButton);
        buttonPanel.add(cancelButton);
        settingsDialog.add(buttonPanel, BorderLayout.SOUTH);

        settingsDialog.pack();
        settingsDialog.setLocationRelativeTo(this);
        settingsDialog.setVisible(true);
    }

    private void onShowAbout() {
        String message = "<html><body style='width: 300px; padding: 10px;'>" +
            "<h2>VTEA Deep Learning</h2>" +
            "<p><b>Version:</b> 1.0</p>" +
            "<p><b>Author:</b> VTEA Deep Learning Team</p>" +
            "<p><b>Description:</b> 3D deep learning classification for volumetric tissue exploration and analysis</p>" +
            "<br>" +
            "<p><b>Features:</b></p>" +
            "<ul>" +
            "<li>NephNet3D and Generic3DCNN architectures</li>" +
            "<li>PyTorch integration via bytedeco</li>" +
            "<li>GPU acceleration support</li>" +
            "<li>Real-time training monitoring</li>" +
            "<li>User-defined class naming</li>" +
            "</ul>" +
            "</body></html>";

        JOptionPane.showMessageDialog(this,
            message,
            "About VTEA Deep Learning",
            JOptionPane.INFORMATION_MESSAGE);
    }

    private void onShowDocumentation() {
        String message = "<html><body style='width: 400px; padding: 10px;'>" +
            "<h2>VTEA Deep Learning Documentation</h2>" +
            "<br>" +
            "<h3>Getting Started:</h3>" +
            "<ol>" +
            "<li><b>Model Configuration:</b> Configure model architecture and parameters</li>" +
            "<li><b>Training:</b> Train models on labeled data with real-time monitoring</li>" +
            "<li><b>Inference:</b> Use trained models to classify new cells</li>" +
            "</ol>" +
            "<br>" +
            "<h3>Supported Models:</h3>" +
            "<ul>" +
            "<li><b>NephNet3D:</b> Optimized for kidney cell classification</li>" +
            "<li><b>Generic3DCNN:</b> Flexible architecture for various tasks</li>" +
            "</ul>" +
            "<br>" +
            "<h3>Workflow:</h3>" +
            "<p>1. Segment cells in VTEA<br>" +
            "2. Use Manual Classification to label training data<br>" +
            "3. Configure and train model<br>" +
            "4. Apply model to classify new data using the Deep Learning Classification plugin</p>" +
            "</body></html>";

        JOptionPane.showMessageDialog(this,
            message,
            "Documentation",
            JOptionPane.INFORMATION_MESSAGE);
    }

    /**
     * Get model configuration panel
     */
    public ModelConfigurationPanel getModelConfigPanel() {
        return modelConfigPanel;
    }

    /**
     * Get training panel
     */
    public TrainingPanel getTrainingPanel() {
        return trainingPanel;
    }

    /**
     * Main method for standalone testing
     */
    public static void main(String[] args) {
        // Set look and feel
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }

        SwingUtilities.invokeLater(() -> {
            DeepLearningUI ui = new DeepLearningUI();
            ui.setVisible(true);
        });
    }
}
