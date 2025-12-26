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
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import vtea.processor.listeners.ProgressListener;

/**
 * Real-time progress visualization panel for VAE training.
 * Displays loss curves, metrics, and progress bars.
 *
 * @author VTEA Developer
 */
public class VAETrainingProgressPanel extends JPanel implements ProgressListener {

    // Progress bars
    private JProgressBar epochProgressBar;
    private JProgressBar batchProgressBar;

    // Status labels
    private JLabel statusLabel;
    private JLabel epochLabel;
    private JLabel trainLossLabel;
    private JLabel valLossLabel;
    private JLabel elboLabel;
    private JLabel reconLossLabel;
    private JLabel klLossLabel;
    private JLabel bestValLossLabel;

    // Charts
    private XYSeries trainLossSeries;
    private XYSeries valLossSeries;
    private XYSeries elboSeries;
    private XYSeries reconLossSeries;
    private XYSeries klLossSeries;
    private ChartPanel lossChartPanel;
    private ChartPanel componentChartPanel;

    // State
    private int currentEpoch = 0;
    private int totalEpochs = 100;
    private double bestValLoss = Double.MAX_VALUE;

    /**
     * Creates a new training progress panel.
     */
    public VAETrainingProgressPanel() {
        initComponents();
    }

    private void initComponents() {
        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Top: Progress bars and status
        add(createProgressPanel(), BorderLayout.NORTH);

        // Center: Charts
        add(createChartsPanel(), BorderLayout.CENTER);

        // Bottom: Metrics
        add(createMetricsPanel(), BorderLayout.SOUTH);
    }

    private JPanel createProgressPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "Training Progress",
            TitledBorder.LEFT, TitledBorder.TOP));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        // Status label
        gbc.gridx = 0; gbc.gridy = 0; gbc.gridwidth = 2;
        statusLabel = new JLabel("Status: Ready");
        statusLabel.setFont(statusLabel.getFont().deriveFont(Font.BOLD, 14f));
        panel.add(statusLabel, gbc);

        // Epoch progress
        gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 1; gbc.weightx = 0;
        epochLabel = new JLabel("Epoch: 0 / 0");
        panel.add(epochLabel, gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        epochProgressBar = new JProgressBar(0, 100);
        epochProgressBar.setStringPainted(true);
        panel.add(epochProgressBar, gbc);

        // Batch progress
        gbc.gridx = 0; gbc.gridy = 2; gbc.weightx = 0;
        panel.add(new JLabel("Batch:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        batchProgressBar = new JProgressBar(0, 100);
        batchProgressBar.setStringPainted(true);
        panel.add(batchProgressBar, gbc);

        return panel;
    }

    private JPanel createChartsPanel() {
        JPanel panel = new JPanel(new GridLayout(1, 2, 10, 10));

        // Loss chart
        trainLossSeries = new XYSeries("Train Loss");
        valLossSeries = new XYSeries("Val Loss");

        XYSeriesCollection lossDataset = new XYSeriesCollection();
        lossDataset.addSeries(trainLossSeries);
        lossDataset.addSeries(valLossSeries);

        JFreeChart lossChart = ChartFactory.createXYLineChart(
            "Total Loss",
            "Epoch",
            "Loss",
            lossDataset,
            PlotOrientation.VERTICAL,
            true, true, false
        );

        customizeChart(lossChart, Color.BLUE, Color.RED);
        lossChartPanel = new ChartPanel(lossChart);
        lossChartPanel.setPreferredSize(new Dimension(400, 300));
        panel.add(lossChartPanel);

        // Component losses chart
        elboSeries = new XYSeries("ELBO");
        reconLossSeries = new XYSeries("Recon Loss");
        klLossSeries = new XYSeries("KL Loss");

        XYSeriesCollection componentDataset = new XYSeriesCollection();
        componentDataset.addSeries(elboSeries);
        componentDataset.addSeries(reconLossSeries);
        componentDataset.addSeries(klLossSeries);

        JFreeChart componentChart = ChartFactory.createXYLineChart(
            "Loss Components",
            "Epoch",
            "Value",
            componentDataset,
            PlotOrientation.VERTICAL,
            true, true, false
        );

        customizeChart(componentChart, Color.GREEN, Color.ORANGE, Color.MAGENTA);
        componentChartPanel = new ChartPanel(componentChart);
        componentChartPanel.setPreferredSize(new Dimension(400, 300));
        panel.add(componentChartPanel);

        return panel;
    }

    private void customizeChart(JFreeChart chart, Color... colors) {
        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        for (int i = 0; i < colors.length && i < plot.getSeriesCount(); i++) {
            renderer.setSeriesPaint(i, colors[i]);
            renderer.setSeriesShapesVisible(i, false);
            renderer.setSeriesStroke(i, new BasicStroke(2.0f));
        }
        plot.setRenderer(renderer);
    }

    private JPanel createMetricsPanel() {
        JPanel panel = new JPanel(new GridLayout(2, 4, 10, 5));
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "Current Metrics",
            TitledBorder.LEFT, TitledBorder.TOP));

        // Train loss
        panel.add(new JLabel("Train Loss:"));
        trainLossLabel = new JLabel("N/A");
        trainLossLabel.setFont(trainLossLabel.getFont().deriveFont(Font.BOLD));
        panel.add(trainLossLabel);

        // Val loss
        panel.add(new JLabel("Val Loss:"));
        valLossLabel = new JLabel("N/A");
        valLossLabel.setFont(valLossLabel.getFont().deriveFont(Font.BOLD));
        panel.add(valLossLabel);

        // ELBO
        panel.add(new JLabel("ELBO:"));
        elboLabel = new JLabel("N/A");
        panel.add(elboLabel);

        // Recon loss
        panel.add(new JLabel("Recon Loss:"));
        reconLossLabel = new JLabel("N/A");
        panel.add(reconLossLabel);

        // KL loss
        panel.add(new JLabel("KL Loss:"));
        klLossLabel = new JLabel("N/A");
        panel.add(klLossLabel);

        // Best val loss
        panel.add(new JLabel("Best Val Loss:"));
        bestValLossLabel = new JLabel("N/A");
        bestValLossLabel.setFont(bestValLossLabel.getFont().deriveFont(Font.BOLD));
        bestValLossLabel.setForeground(new Color(0, 128, 0));
        panel.add(bestValLossLabel);

        return panel;
    }

    /**
     * Updates progress for current epoch.
     *
     * @param epoch Current epoch number
     * @param totalEpochs Total number of epochs
     */
    public void updateEpochProgress(int epoch, int totalEpochs) {
        this.currentEpoch = epoch;
        this.totalEpochs = totalEpochs;

        SwingUtilities.invokeLater(() -> {
            epochLabel.setText(String.format("Epoch: %d / %d", epoch, totalEpochs));
            int progress = (int) ((epoch / (double) totalEpochs) * 100);
            epochProgressBar.setValue(progress);
        });
    }

    /**
     * Updates progress for current batch.
     *
     * @param batch Current batch number
     * @param totalBatches Total number of batches
     */
    public void updateBatchProgress(int batch, int totalBatches) {
        SwingUtilities.invokeLater(() -> {
            int progress = (int) ((batch / (double) totalBatches) * 100);
            batchProgressBar.setValue(progress);
            batchProgressBar.setString(String.format("%d / %d", batch, totalBatches));
        });
    }

    /**
     * Updates loss metrics and charts.
     *
     * @param epoch Current epoch
     * @param trainLoss Training loss
     * @param valLoss Validation loss
     * @param elbo ELBO value
     * @param reconLoss Reconstruction loss
     * @param klLoss KL divergence loss
     */
    public void updateMetrics(int epoch, double trainLoss, double valLoss,
                             double elbo, double reconLoss, double klLoss) {
        SwingUtilities.invokeLater(() -> {
            // Update labels
            trainLossLabel.setText(String.format("%.6f", trainLoss));
            valLossLabel.setText(String.format("%.6f", valLoss));
            elboLabel.setText(String.format("%.6f", elbo));
            reconLossLabel.setText(String.format("%.6f", reconLoss));
            klLossLabel.setText(String.format("%.6f", klLoss));

            // Update best val loss
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                bestValLossLabel.setText(String.format("%.6f ⭐", bestValLoss));
            }

            // Update charts
            trainLossSeries.add(epoch, trainLoss);
            valLossSeries.add(epoch, valLoss);
            elboSeries.add(epoch, elbo);
            reconLossSeries.add(epoch, reconLoss);
            klLossSeries.add(epoch, klLoss);
        });
    }

    /**
     * Sets the status message.
     *
     * @param status Status message
     */
    public void setStatus(String status) {
        SwingUtilities.invokeLater(() -> {
            statusLabel.setText("Status: " + status);
        });
    }

    /**
     * Resets all metrics and charts.
     */
    public void reset() {
        SwingUtilities.invokeLater(() -> {
            currentEpoch = 0;
            bestValLoss = Double.MAX_VALUE;

            epochProgressBar.setValue(0);
            batchProgressBar.setValue(0);

            trainLossLabel.setText("N/A");
            valLossLabel.setText("N/A");
            elboLabel.setText("N/A");
            reconLossLabel.setText("N/A");
            klLossLabel.setText("N/A");
            bestValLossLabel.setText("N/A");

            trainLossSeries.clear();
            valLossSeries.clear();
            elboSeries.clear();
            reconLossSeries.clear();
            klLossSeries.clear();

            statusLabel.setText("Status: Ready");
        });
    }

    // ProgressListener implementation
    @Override
    public void sendProgress(double progress) {
        // Not used - we have more specific update methods
    }

    @Override
    public void sendProgress(double progress, int total) {
        SwingUtilities.invokeLater(() -> {
            int percentage = (int) ((progress / total) * 100);
            epochProgressBar.setValue(percentage);
        });
    }
}
