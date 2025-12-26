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
import java.awt.*;

/**
 * Composite training panel that can be embedded in VTEA workflows.
 * Combines configuration, data selection, and progress in a single panel.
 *
 * @author VTEA Developer
 */
public class VAETrainingPanel extends JPanel {

    private VAEConfigurationPanel configPanel;
    private VAEDataSelectionPanel dataPanel;
    private VAETrainingProgressPanel progressPanel;
    private JButton launchDialogButton;

    /**
     * Creates a new VAE training panel.
     */
    public VAETrainingPanel() {
        this(false);
    }

    /**
     * Creates a new VAE training panel.
     *
     * @param embedded If true, shows compact embedded view; if false, shows full view
     */
    public VAETrainingPanel(boolean embedded) {
        initComponents(embedded);
    }

    private void initComponents(boolean embedded) {
        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        if (embedded) {
            // Compact embedded view - just show launch button
            JPanel panel = new JPanel(new GridBagLayout());
            GridBagConstraints gbc = new GridBagConstraints();
            gbc.insets = new Insets(10, 10, 10, 10);

            JLabel label = new JLabel("VAE Model Training");
            label.setFont(label.getFont().deriveFont(Font.BOLD, 16f));
            gbc.gridx = 0; gbc.gridy = 0;
            panel.add(label, gbc);

            launchDialogButton = new JButton("Open Training Dialog");
            launchDialogButton.setPreferredSize(new Dimension(200, 40));
            launchDialogButton.setFont(launchDialogButton.getFont().deriveFont(Font.BOLD));
            launchDialogButton.addActionListener(e -> launchDialog());
            gbc.gridy = 1;
            panel.add(launchDialogButton, gbc);

            JLabel description = new JLabel("<html><center>Train 3D VAE models for<br>" +
                                           "feature extraction and analysis</center></html>");
            description.setForeground(Color.GRAY);
            gbc.gridy = 2;
            panel.add(description, gbc);

            add(panel, BorderLayout.CENTER);

        } else {
            // Full view - show all panels in tabs
            JTabbedPane tabbedPane = new JTabbedPane();

            configPanel = new VAEConfigurationPanel();
            JScrollPane configScroll = new JScrollPane(configPanel);
            configScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
            tabbedPane.addTab("Configuration", configScroll);

            dataPanel = new VAEDataSelectionPanel();
            JScrollPane dataScroll = new JScrollPane(dataPanel);
            dataScroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
            tabbedPane.addTab("Data", dataScroll);

            progressPanel = new VAETrainingProgressPanel();
            tabbedPane.addTab("Progress", progressPanel);

            add(tabbedPane, BorderLayout.CENTER);

            // Control buttons
            add(createControlPanel(), BorderLayout.SOUTH);
        }
    }

    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 10));

        JButton launchButton = new JButton("Launch Training Dialog");
        launchButton.addActionListener(e -> launchDialog());
        panel.add(launchButton);

        return panel;
    }

    private void launchDialog() {
        VAETrainingDialog dialog = new VAETrainingDialog();

        // If we have config panel, transfer settings
        if (configPanel != null) {
            dialog.getConfigPanel().setConfig(configPanel.getConfig());
            dialog.getConfigPanel().setCheckpointDir(configPanel.getCheckpointDir());
        }

        dialog.showDialog();
    }

    /**
     * Gets the configuration panel.
     *
     * @return Configuration panel or null if in embedded mode
     */
    public VAEConfigurationPanel getConfigPanel() {
        return configPanel;
    }

    /**
     * Gets the data selection panel.
     *
     * @return Data selection panel or null if in embedded mode
     */
    public VAEDataSelectionPanel getDataPanel() {
        return dataPanel;
    }

    /**
     * Gets the progress panel.
     *
     * @return Progress panel or null if in embedded mode
     */
    public VAETrainingProgressPanel getProgressPanel() {
        return progressPanel;
    }

    /**
     * Creates a standalone frame with this panel.
     *
     * @return JFrame containing the training panel
     */
    public static JFrame createFrame() {
        JFrame frame = new JFrame("VAE Training");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setContentPane(new VAETrainingPanel(false));
        frame.setSize(900, 700);
        frame.setLocationRelativeTo(null);
        return frame;
    }

    /**
     * Main method for standalone testing.
     *
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = createFrame();
            frame.setVisible(true);
        });
    }
}
