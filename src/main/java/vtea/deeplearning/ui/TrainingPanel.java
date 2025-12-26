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

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import vtea.deeplearning.training.Metrics;
import vtea.deeplearning.training.TrainingConfig;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Panel for configuring and monitoring model training.
 * Includes real-time progress visualization and configuration options.
 *
 * @author VTEA Deep Learning Team
 */
public class TrainingPanel extends JPanel {

    // Training configuration
    private JSpinner epochsSpinner;
    private JSpinner batchSizeSpinner;
    private JSpinner learningRateSpinner;
    private JComboBox<String> optimizerCombo;
    private JSpinner weightDecaySpinner;
    private JSpinner validationSplitSpinner;

    // Advanced configuration
    private JCheckBox useEarlyStoppingCheckbox;
    private JSpinner earlyStoppingPatienceSpinner;
    private JCheckBox useLRSchedulerCheckbox;
    private JComboBox<String> lrSchedulerTypeCombo;
    private JSpinner lrDecayEpochsSpinner;

    // Progress monitoring
    private JProgressBar epochProgressBar;
    private JProgressBar batchProgressBar;
    private JLabel currentEpochLabel;
    private JLabel currentLossLabel;
    private JLabel currentAccuracyLabel;
    private JLabel valAccuracyLabel;
    private JLabel balancedAccuracyLabel;
    private JLabel etaLabel;

    // Training history chart
    private XYSeries trainLossSeries;
    private XYSeries valLossSeries;
    private XYSeries trainAccSeries;
    private XYSeries valAccSeries;
    private XYSeries valBalAccSeries;
    private ChartPanel lossChartPanel;
    private ChartPanel accuracyChartPanel;

    // Control buttons
    private JButton startButton;
    private JButton pauseButton;
    private JButton stopButton;

    // Training state
    private boolean isTraining = false;
    private long trainingStartTime;
    private int totalEpochs;

    /**
     * Constructor
     */
    public TrainingPanel() {
        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Create main split pane
        JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
        splitPane.setResizeWeight(0.4);

        // Top: Configuration
        JPanel configPanel = new JPanel(new BorderLayout());
        JTabbedPane configTabs = new JTabbedPane();
        configTabs.addTab("Basic", createBasicConfigPanel());
        configTabs.addTab("Advanced", createAdvancedConfigPanel());
        configPanel.add(configTabs, BorderLayout.CENTER);

        splitPane.setTopComponent(new JScrollPane(configPanel));

        // Bottom: Progress monitoring
        JPanel progressPanel = new JPanel(new BorderLayout());
        progressPanel.add(createProgressSection(), BorderLayout.NORTH);
        progressPanel.add(createChartsSection(), BorderLayout.CENTER);
        progressPanel.add(createControlPanel(), BorderLayout.SOUTH);

        splitPane.setBottomComponent(progressPanel);

        add(splitPane, BorderLayout.CENTER);

        // Initialize charts
        initializeCharts();
    }

    /**
     * Create basic configuration panel
     */
    private JPanel createBasicConfigPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;

        int row = 0;

        // Epochs
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Epochs:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        epochsSpinner = new JSpinner(new SpinnerNumberModel(100, 1, 1000, 10));
        panel.add(epochsSpinner, gbc);

        row++;

        // Batch size
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0.0;
        panel.add(new JLabel("Batch Size:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        batchSizeSpinner = new JSpinner(new SpinnerNumberModel(16, 1, 128, 1));
        panel.add(batchSizeSpinner, gbc);

        row++;

        // Learning rate
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0.0;
        panel.add(new JLabel("Learning Rate:"), gbc);

        gbc.gridx = 1;
        learningRateSpinner = new JSpinner(new SpinnerNumberModel(0.001, 0.00001, 0.1, 0.0001));
        JSpinner.NumberEditor editor = new JSpinner.NumberEditor(learningRateSpinner, "0.######");
        learningRateSpinner.setEditor(editor);
        panel.add(learningRateSpinner, gbc);

        row++;

        // Optimizer
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Optimizer:"), gbc);

        gbc.gridx = 1;
        optimizerCombo = new JComboBox<>(new String[]{"ADAM", "ADAMW", "SGD", "RMSPROP"});
        panel.add(optimizerCombo, gbc);

        row++;

        // Weight decay
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Weight Decay:"), gbc);

        gbc.gridx = 1;
        weightDecaySpinner = new JSpinner(new SpinnerNumberModel(0.00001, 0.0, 0.01, 0.00001));
        JSpinner.NumberEditor wdEditor = new JSpinner.NumberEditor(weightDecaySpinner, "0.######");
        weightDecaySpinner.setEditor(wdEditor);
        panel.add(weightDecaySpinner, gbc);

        row++;

        // Validation split
        gbc.gridx = 0;
        gbc.gridy = row;
        panel.add(new JLabel("Validation Split:"), gbc);

        gbc.gridx = 1;
        validationSplitSpinner = new JSpinner(new SpinnerNumberModel(0.2, 0.1, 0.5, 0.05));
        JSpinner.NumberEditor vsEditor = new JSpinner.NumberEditor(validationSplitSpinner, "0.##");
        validationSplitSpinner.setEditor(vsEditor);
        panel.add(validationSplitSpinner, gbc);

        return panel;
    }

    /**
     * Create advanced configuration panel
     */
    private JPanel createAdvancedConfigPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;

        int row = 0;

        // Early stopping
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.gridwidth = 2;
        useEarlyStoppingCheckbox = new JCheckBox("Use Early Stopping", true);
        useEarlyStoppingCheckbox.addActionListener(e ->
            earlyStoppingPatienceSpinner.setEnabled(useEarlyStoppingCheckbox.isSelected()));
        panel.add(useEarlyStoppingCheckbox, gbc);

        row++;

        // Early stopping patience
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.gridwidth = 1;
        gbc.weightx = 0.0;
        panel.add(new JLabel("  Patience:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        earlyStoppingPatienceSpinner = new JSpinner(new SpinnerNumberModel(10, 1, 50, 1));
        panel.add(earlyStoppingPatienceSpinner, gbc);

        row++;

        // Learning rate scheduler
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.gridwidth = 2;
        gbc.weightx = 0.0;
        useLRSchedulerCheckbox = new JCheckBox("Use Learning Rate Scheduler", true);
        useLRSchedulerCheckbox.addActionListener(e -> {
            lrSchedulerTypeCombo.setEnabled(useLRSchedulerCheckbox.isSelected());
            lrDecayEpochsSpinner.setEnabled(useLRSchedulerCheckbox.isSelected());
        });
        panel.add(useLRSchedulerCheckbox, gbc);

        row++;

        // LR scheduler type
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.gridwidth = 1;
        panel.add(new JLabel("  Scheduler Type:"), gbc);

        gbc.gridx = 1;
        gbc.weightx = 1.0;
        lrSchedulerTypeCombo = new JComboBox<>(new String[]{"STEP", "EXPONENTIAL", "COSINE"});
        panel.add(lrSchedulerTypeCombo, gbc);

        row++;

        // LR decay epochs
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0.0;
        panel.add(new JLabel("  Decay Epochs:"), gbc);

        gbc.gridx = 1;
        lrDecayEpochsSpinner = new JSpinner(new SpinnerNumberModel(30, 1, 100, 5));
        panel.add(lrDecayEpochsSpinner, gbc);

        return panel;
    }

    /**
     * Create progress monitoring section
     */
    private JPanel createProgressSection() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(new TitledBorder("Training Progress"));

        // Epoch progress
        JPanel epochPanel = new JPanel(new BorderLayout(5, 5));
        currentEpochLabel = new JLabel("Epoch: 0/0");
        epochPanel.add(currentEpochLabel, BorderLayout.WEST);
        epochProgressBar = new JProgressBar(0, 100);
        epochProgressBar.setStringPainted(true);
        epochPanel.add(epochProgressBar, BorderLayout.CENTER);
        panel.add(epochPanel);

        // Batch progress
        JPanel batchPanel = new JPanel(new BorderLayout(5, 5));
        batchPanel.add(new JLabel("Batch:"), BorderLayout.WEST);
        batchProgressBar = new JProgressBar(0, 100);
        batchProgressBar.setStringPainted(true);
        batchPanel.add(batchProgressBar, BorderLayout.CENTER);
        panel.add(batchPanel);

        // Metrics panel
        JPanel metricsPanel = new JPanel(new GridLayout(2, 3, 10, 5));
        metricsPanel.setBorder(BorderFactory.createEmptyBorder(5, 0, 0, 0));

        currentLossLabel = new JLabel("Loss: --");
        currentAccuracyLabel = new JLabel("Acc: --");
        valAccuracyLabel = new JLabel("Val Acc: --");
        balancedAccuracyLabel = new JLabel("Bal Acc: --");
        etaLabel = new JLabel("ETA: --");

        metricsPanel.add(currentLossLabel);
        metricsPanel.add(currentAccuracyLabel);
        metricsPanel.add(valAccuracyLabel);
        metricsPanel.add(balancedAccuracyLabel);
        metricsPanel.add(etaLabel);
        metricsPanel.add(new JLabel(""));  // Spacer

        panel.add(metricsPanel);

        return panel;
    }

    /**
     * Create charts section
     */
    private JPanel createChartsSection() {
        JPanel panel = new JPanel(new GridLayout(1, 2, 10, 0));

        // Initialize data series
        trainLossSeries = new XYSeries("Train Loss");
        valLossSeries = new XYSeries("Val Loss");

        XYSeriesCollection lossDataset = new XYSeriesCollection();
        lossDataset.addSeries(trainLossSeries);
        lossDataset.addSeries(valLossSeries);

        JFreeChart lossChart = ChartFactory.createXYLineChart(
            "Loss",
            "Epoch",
            "Loss",
            lossDataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        lossChartPanel = new ChartPanel(lossChart);
        lossChartPanel.setPreferredSize(new Dimension(400, 300));
        panel.add(lossChartPanel);

        // Accuracy chart
        trainAccSeries = new XYSeries("Train Acc");
        valAccSeries = new XYSeries("Val Acc");
        valBalAccSeries = new XYSeries("Val Bal Acc");

        XYSeriesCollection accDataset = new XYSeriesCollection();
        accDataset.addSeries(trainAccSeries);
        accDataset.addSeries(valAccSeries);
        accDataset.addSeries(valBalAccSeries);

        JFreeChart accChart = ChartFactory.createXYLineChart(
            "Accuracy",
            "Epoch",
            "Accuracy",
            accDataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        accuracyChartPanel = new ChartPanel(accChart);
        accuracyChartPanel.setPreferredSize(new Dimension(400, 300));
        panel.add(accuracyChartPanel);

        return panel;
    }

    /**
     * Create control panel
     */
    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.RIGHT));

        startButton = new JButton("Start Training");
        startButton.addActionListener(e -> onStartTraining());

        pauseButton = new JButton("Pause");
        pauseButton.setEnabled(false);
        pauseButton.addActionListener(e -> onPauseTraining());

        stopButton = new JButton("Stop");
        stopButton.setEnabled(false);
        stopButton.addActionListener(e -> onStopTraining());

        panel.add(startButton);
        panel.add(pauseButton);
        panel.add(stopButton);

        return panel;
    }

    /**
     * Initialize charts
     */
    private void initializeCharts() {
        trainLossSeries.clear();
        valLossSeries.clear();
        trainAccSeries.clear();
        valAccSeries.clear();
        valBalAccSeries.clear();
    }

    /**
     * Update progress from training
     */
    public void updateProgress(int epoch, int totalEpochs, int batch, int totalBatches,
                               double trainLoss, double trainAcc,
                               double valLoss, double valAcc, double valBalAcc) {
        SwingUtilities.invokeLater(() -> {
            // Update epoch progress
            currentEpochLabel.setText(String.format("Epoch: %d/%d", epoch + 1, totalEpochs));
            int epochProgress = (int) (100.0 * (epoch + 1) / totalEpochs);
            epochProgressBar.setValue(epochProgress);

            // Update batch progress
            int batchProgress = (int) (100.0 * batch / totalBatches);
            batchProgressBar.setValue(batchProgress);

            // Update metrics
            currentLossLabel.setText(String.format("Loss: %.4f", trainLoss));
            currentAccuracyLabel.setText(String.format("Acc: %.4f", trainAcc));
            valAccuracyLabel.setText(String.format("Val Acc: %.4f", valAcc));
            balancedAccuracyLabel.setText(String.format("Bal Acc: %.4f", valBalAcc));

            // Update ETA
            if (trainingStartTime > 0) {
                long elapsed = System.currentTimeMillis() - trainingStartTime;
                long totalEstimated = elapsed * totalEpochs / (epoch + 1);
                long remaining = totalEstimated - elapsed;
                etaLabel.setText(String.format("ETA: %s", formatTime(remaining)));
            }

            // Update charts
            trainLossSeries.addOrUpdate(epoch + 1, trainLoss);
            valLossSeries.addOrUpdate(epoch + 1, valLoss);
            trainAccSeries.addOrUpdate(epoch + 1, trainAcc);
            valAccSeries.addOrUpdate(epoch + 1, valAcc);
            valBalAccSeries.addOrUpdate(epoch + 1, valBalAcc);
        });
    }

    /**
     * Format time in milliseconds to human-readable string
     */
    private String formatTime(long millis) {
        long seconds = millis / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;

        if (hours > 0) {
            return String.format("%dh %dm", hours, minutes % 60);
        } else if (minutes > 0) {
            return String.format("%dm %ds", minutes, seconds % 60);
        } else {
            return String.format("%ds", seconds);
        }
    }

    /**
     * Get training configuration
     */
    public TrainingConfig getTrainingConfig() {
        return new TrainingConfig.Builder()
            .epochs((Integer) epochsSpinner.getValue())
            .batchSize((Integer) batchSizeSpinner.getValue())
            .learningRate((Double) learningRateSpinner.getValue())
            .optimizer(TrainingConfig.OptimizerType.valueOf((String) optimizerCombo.getSelectedItem()))
            .weightDecay((Double) weightDecaySpinner.getValue())
            .validationSplit((Double) validationSplitSpinner.getValue())
            .useEarlyStopping(useEarlyStoppingCheckbox.isSelected())
            .earlyStoppingPatience((Integer) earlyStoppingPatienceSpinner.getValue())
            .verbose(true)
            .build();
    }

    /**
     * Event handlers
     */
    private void onStartTraining() {
        isTraining = true;
        trainingStartTime = System.currentTimeMillis();
        totalEpochs = (Integer) epochsSpinner.getValue();

        startButton.setEnabled(false);
        pauseButton.setEnabled(true);
        stopButton.setEnabled(true);

        // Disable configuration during training
        setConfigEnabled(false);

        // Reset charts
        initializeCharts();
    }

    private void onPauseTraining() {
        isTraining = false;
        pauseButton.setText(isTraining ? "Pause" : "Resume");
    }

    private void onStopTraining() {
        isTraining = false;

        startButton.setEnabled(true);
        pauseButton.setEnabled(false);
        stopButton.setEnabled(false);

        // Re-enable configuration
        setConfigEnabled(true);
    }

    private void setConfigEnabled(boolean enabled) {
        epochsSpinner.setEnabled(enabled);
        batchSizeSpinner.setEnabled(enabled);
        learningRateSpinner.setEnabled(enabled);
        optimizerCombo.setEnabled(enabled);
        weightDecaySpinner.setEnabled(enabled);
        validationSplitSpinner.setEnabled(enabled);
        useEarlyStoppingCheckbox.setEnabled(enabled);
        earlyStoppingPatienceSpinner.setEnabled(enabled);
        useLRSchedulerCheckbox.setEnabled(enabled);
        lrSchedulerTypeCombo.setEnabled(enabled);
        lrDecayEpochsSpinner.setEnabled(enabled);
    }

    /**
     * Test main
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Training Panel");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            TrainingPanel panel = new TrainingPanel();
            frame.add(panel);

            // Simulate training progress
            new Thread(() -> {
                panel.onStartTraining();
                for (int epoch = 0; epoch < 100; epoch++) {
                    for (int batch = 0; batch < 50; batch++) {
                        panel.updateProgress(epoch, 100, batch, 50,
                            2.0 - epoch * 0.01 + Math.random() * 0.1,
                            epoch * 0.005 + Math.random() * 0.05,
                            2.5 - epoch * 0.01 + Math.random() * 0.1,
                            epoch * 0.004 + Math.random() * 0.05,
                            epoch * 0.004 + Math.random() * 0.05);

                        try {
                            Thread.sleep(50);
                        } catch (InterruptedException e) {
                            break;
                        }
                    }
                }
                panel.onStopTraining();
            }).start();

            frame.setSize(900, 800);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
