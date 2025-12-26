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

import ij.ImagePlus;
import ij.WindowManager;
import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import vtea.objects.MicroObject;
import vtea.protocol.setup.MicroBlockSetup;

/**
 * Panel for selecting training data sources.
 * Allows selection of image stacks and segmented objects.
 *
 * @author VTEA Developer
 */
public class VAEDataSelectionPanel extends JPanel {

    private JComboBox<String> imageStackCombo;
    private JRadioButton useOpenImagesRadio;
    private JRadioButton useFileRadio;
    private JTextField filePathField;
    private JButton browseFileButton;

    private JRadioButton useCurrentObjectsRadio;
    private JRadioButton useAllObjectsRadio;
    private JLabel objectCountLabel;

    private JSpinner trainSplitSpinner;
    private JCheckBox useAugmentationCheckbox;

    /**
     * Creates a new data selection panel.
     */
    public VAEDataSelectionPanel() {
        initComponents();
    }

    private void initComponents() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Image source panel
        add(createImageSourcePanel());
        add(Box.createVerticalStrut(10));

        // Object source panel
        add(createObjectSourcePanel());
        add(Box.createVerticalStrut(10));

        // Training options panel
        add(createTrainingOptionsPanel());
    }

    private JPanel createImageSourcePanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "Image Data Source",
            TitledBorder.LEFT, TitledBorder.TOP));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        // Radio button group
        ButtonGroup sourceGroup = new ButtonGroup();

        // Option 1: Use open images
        gbc.gridx = 0; gbc.gridy = 0; gbc.gridwidth = 3;
        useOpenImagesRadio = new JRadioButton("Use Open ImageJ Images", true);
        useOpenImagesRadio.addActionListener(e -> onSourceChanged());
        sourceGroup.add(useOpenImagesRadio);
        panel.add(useOpenImagesRadio, gbc);

        // Image stack selector
        gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 1; gbc.weightx = 0;
        panel.add(Box.createHorizontalStrut(30), gbc);

        gbc.gridx = 1; gbc.weightx = 0;
        panel.add(new JLabel("Image:"), gbc);

        gbc.gridx = 2; gbc.weightx = 1.0;
        imageStackCombo = new JComboBox<>();
        updateImageList();
        panel.add(imageStackCombo, gbc);

        // Option 2: Use file
        gbc.gridx = 0; gbc.gridy = 2; gbc.gridwidth = 3; gbc.weightx = 0;
        useFileRadio = new JRadioButton("Use Image File");
        useFileRadio.addActionListener(e -> onSourceChanged());
        sourceGroup.add(useFileRadio);
        panel.add(useFileRadio, gbc);

        // File path
        gbc.gridx = 0; gbc.gridy = 3; gbc.gridwidth = 1;
        panel.add(Box.createHorizontalStrut(30), gbc);

        gbc.gridx = 1; gbc.weightx = 0;
        panel.add(new JLabel("File:"), gbc);

        gbc.gridx = 2; gbc.weightx = 1.0;
        filePathField = new JTextField();
        filePathField.setEnabled(false);
        panel.add(filePathField, gbc);

        gbc.gridx = 3; gbc.weightx = 0;
        browseFileButton = new JButton("Browse...");
        browseFileButton.setEnabled(false);
        browseFileButton.addActionListener(e -> browseImageFile());
        panel.add(browseFileButton, gbc);

        return panel;
    }

    private JPanel createObjectSourcePanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "Segmented Objects",
            TitledBorder.LEFT, TitledBorder.TOP));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        ButtonGroup objectGroup = new ButtonGroup();

        // Option 1: Use current objects from MicroBlockSetup
        gbc.gridx = 0; gbc.gridy = 0; gbc.gridwidth = 2;
        useCurrentObjectsRadio = new JRadioButton("Use Current Segmented Objects", true);
        useCurrentObjectsRadio.addActionListener(e -> updateObjectCount());
        objectGroup.add(useCurrentObjectsRadio);
        panel.add(useCurrentObjectsRadio, gbc);

        // Option 2: Use all objects
        gbc.gridx = 0; gbc.gridy = 1;
        useAllObjectsRadio = new JRadioButton("Use All Available Objects");
        useAllObjectsRadio.addActionListener(e -> updateObjectCount());
        objectGroup.add(useAllObjectsRadio);
        panel.add(useAllObjectsRadio, gbc);

        // Object count
        gbc.gridx = 0; gbc.gridy = 2; gbc.gridwidth = 1; gbc.weightx = 0;
        panel.add(new JLabel("Objects Available:"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        objectCountLabel = new JLabel("0");
        objectCountLabel.setFont(objectCountLabel.getFont().deriveFont(Font.BOLD));
        panel.add(objectCountLabel, gbc);

        updateObjectCount();

        return panel;
    }

    private JPanel createTrainingOptionsPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), "Data Options",
            TitledBorder.LEFT, TitledBorder.TOP));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;

        // Train/val split
        gbc.gridx = 0; gbc.gridy = 0;
        panel.add(new JLabel("Train/Val Split (%):"), gbc);

        gbc.gridx = 1; gbc.weightx = 1.0;
        trainSplitSpinner = new JSpinner(new SpinnerNumberModel(80, 50, 95, 5));
        panel.add(trainSplitSpinner, gbc);

        // Data augmentation
        gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 2; gbc.weightx = 0;
        useAugmentationCheckbox = new JCheckBox("Use Data Augmentation (rotation, flip)", true);
        panel.add(useAugmentationCheckbox, gbc);

        return panel;
    }

    private void onSourceChanged() {
        boolean useOpen = useOpenImagesRadio.isSelected();
        imageStackCombo.setEnabled(useOpen);
        filePathField.setEnabled(!useOpen);
        browseFileButton.setEnabled(!useOpen);
    }

    private void browseImageFile() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            @Override
            public boolean accept(File f) {
                if (f.isDirectory()) return true;
                String name = f.getName().toLowerCase();
                return name.endsWith(".tif") || name.endsWith(".tiff");
            }

            @Override
            public String getDescription() {
                return "TIFF Images (*.tif, *.tiff)";
            }
        });

        int result = chooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            filePathField.setText(chooser.getSelectedFile().getAbsolutePath());
        }
    }

    private void updateImageList() {
        imageStackCombo.removeAllItems();

        int[] imageIDs = WindowManager.getIDList();
        if (imageIDs != null) {
            for (int id : imageIDs) {
                ImagePlus imp = WindowManager.getImage(id);
                if (imp != null && imp.getStackSize() > 1) {
                    imageStackCombo.addItem(imp.getTitle());
                }
            }
        }

        if (imageStackCombo.getItemCount() == 0) {
            imageStackCombo.addItem("No images available");
        }
    }

    private void updateObjectCount() {
        try {
            ArrayList<MicroObject> objects = MicroBlockSetup.getMicroObjects();
            int count = (objects != null) ? objects.size() : 0;
            objectCountLabel.setText(String.valueOf(count));
        } catch (Exception e) {
            objectCountLabel.setText("0");
        }
    }

    /**
     * Refreshes the list of available images.
     */
    public void refreshImageList() {
        updateImageList();
        updateObjectCount();
    }

    /**
     * Gets the selected image stack.
     *
     * @return Selected ImagePlus or null if not available
     */
    public ImagePlus getSelectedImage() {
        if (useOpenImagesRadio.isSelected()) {
            String title = (String) imageStackCombo.getSelectedItem();
            if (title != null && !title.equals("No images available")) {
                return WindowManager.getImage(title);
            }
        } else if (useFileRadio.isSelected()) {
            String path = filePathField.getText();
            if (!path.isEmpty() && new File(path).exists()) {
                return new ImagePlus(path);
            }
        }
        return null;
    }

    /**
     * Gets the segmented objects for training.
     *
     * @return List of MicroObjects or null if not available
     */
    public List<MicroObject> getObjects() {
        try {
            return MicroBlockSetup.getMicroObjects();
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Gets the train/validation split percentage.
     *
     * @return Train percentage (e.g., 80 for 80/20 split)
     */
    public int getTrainSplitPercentage() {
        return (int) trainSplitSpinner.getValue();
    }

    /**
     * Checks if data augmentation should be used.
     *
     * @return true if augmentation enabled
     */
    public boolean isAugmentationEnabled() {
        return useAugmentationCheckbox.isSelected();
    }

    /**
     * Validates that all required data is available.
     *
     * @return true if data is valid
     */
    public boolean validateData() {
        ImagePlus imp = getSelectedImage();
        if (imp == null) {
            JOptionPane.showMessageDialog(this,
                "No image selected or image not available.",
                "Data Validation Error",
                JOptionPane.ERROR_MESSAGE);
            return false;
        }

        List<MicroObject> objects = getObjects();
        if (objects == null || objects.isEmpty()) {
            JOptionPane.showMessageDialog(this,
                "No segmented objects available. Please segment objects first.",
                "Data Validation Error",
                JOptionPane.ERROR_MESSAGE);
            return false;
        }

        return true;
    }
}
