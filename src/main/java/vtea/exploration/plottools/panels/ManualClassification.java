/*
 * Copyright (C) 2020 SciJava
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
package vtea.exploration.plottools.panels;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.plugin.Duplicator;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.ListIterator;
import java.util.Random;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JColorChooser;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.filechooser.FileNameExtensionFilter;
import vtea.deeplearning.data.ClassDefinition;
import vtea.deeplearning.data.DatasetDefinition;
import vtea.exploration.listeners.AddFeaturesListener;
import vteaobjects.MicroObject;

/**
 *
 * @author Seth Winfree
 */
public class ManualClassification implements WindowListener{

    ImagePlus image;
    ImagePlus classifying;
    protected ArrayList<MicroObject> objects = new ArrayList();
    protected ArrayList<ArrayList<Number>> measurements = new ArrayList();
    String key;
    int nClasses = 1;
    int nCells = 10;
    int currentCell;
    int count;

    HashMap<Double, Integer> result = new HashMap();
    ArrayList<AddFeaturesListener> addfeaturelisteners = new ArrayList<AddFeaturesListener>();

    ArrayList<JTextField> textFields = new ArrayList<JTextField>();
    ArrayList<JButton> colorButtons = new ArrayList<JButton>();
    ArrayList<JLabel> textLabels = new ArrayList<JLabel>();
    ArrayList<Integer> labelCounts = new ArrayList<Integer>();

    // Class definitions for deep learning integration
    HashMap<Integer, ClassDefinition> classDefinitions = new HashMap<>();

    JFrame classLoggerFrame;

    JLabel total = new JLabel("0", JLabel.CENTER);

    ManualClassification(ImagePlus image, ArrayList objects, ArrayList measurements, String key) {
        this.image = image;
        this.objects = objects;
        this.measurements = measurements;
        this.key = key;   
        this.nCells = objects.size();
    }

    public void process() {
        int[] settings = new int[2];

        SetupManualClassification smc = new SetupManualClassification();

        int result = smc.showDialog();

        if (result == JOptionPane.OK_OPTION) {
            settings = smc.getInformation();
            nClasses = settings[0];
            nCells = settings[1];
            
            ClassLoggerWindow clw = new ClassLoggerWindow(nClasses);
            
            classLoggerFrame = new JFrame();
            
            classLoggerFrame.addWindowListener(this);
            classLoggerFrame.setTitle("Manual Classification Tool");
            classLoggerFrame.add(clw);
            classLoggerFrame.repaint();
            classLoggerFrame.pack();
            classLoggerFrame.setVisible(true);
            
            

            processObject();
        }

    }

    public void processObject() {

        BufferedImage placeholder = new BufferedImage(512,
                512, BufferedImage.TYPE_INT_ARGB);

        Overlay overlay = new Overlay();

        ImageRoi ir = new ImageRoi(0, 0, placeholder);

        Random rand = new Random();


        if (count < nCells) {
            currentCell = rand.nextInt(this.objects.size() - 1);
            
            //System.out.println("PROFILING: Analyzing the " + count +" object: " + currentCell);
            
            MicroObject vol = objects.get(currentCell);

            double ObjectID = vol.getSerialID();


            
            int OffsetX = vol.getBoundingRectangle().x-25;
            int OffsetY = vol.getBoundingRectangle().y-25;

            
            image.setRoi(OffsetX, OffsetY, 50, 50);
            
            Duplicator dup = new Duplicator();

            classifying = dup.run(image);
            
            BufferedImage selections = new BufferedImage(classifying.getWidth(),
                    classifying.getHeight(), BufferedImage.TYPE_INT_ARGB);

            Graphics2D g2 = selections.createGraphics();

            int i = (int)vol.getCentroidZ();
            

            int[] x_pixels = vol.getXPixelsInRegion(i);
            int[] y_pixels = vol.getYPixelsInRegion(i);
            for (int c = 0; c < x_pixels.length; c++) {
                g2.setColor(Color.red);
                g2.drawRect(x_pixels[c]-OffsetX, y_pixels[c]-OffsetY, 1, 1);
            }

            classifying.setTitle("Classifying cell " + ObjectID + " at (" + OffsetX + ", " + OffsetY + ")");
            
            ir = new ImageRoi(0, 0, selections);
            ir.setPosition(0, i , 0);
            ir.setOpacity(0.4);
            overlay.selectable(false);
            overlay.add(ir);
            
            

            classifying.setOverlay(overlay);
            classifying.draw();

            if (classifying.getDisplayMode() != IJ.COMPOSITE) {
                classifying.setDisplayMode(IJ.COMPOSITE);
            }
            classifying.setZ(i);
            classifying.show();
            int Xpos = (int)classifying.getWidth()/2;
            int Ypos = (int)classifying.getHeight()/2;
            IJ.run("Maximize", "");
            count++;
        } else {
            
        }
    }
    
    protected void addFeature(){

        ArrayList<ArrayList<Number>> paddedTable = new ArrayList();
        ArrayList<Number> r = new ArrayList();
        
        for(int i = 0; i < objects.size(); i++){
            
            
            MicroObject m = objects.get(i);
       
            
           result.putIfAbsent(m.getSerialID(), -1);
           r.add(result.get(m.getSerialID()));
          
            }
        
        Random rand = new Random();
 
        paddedTable.add(r);
        notifyAddFeatureListener("Manual_Classification_" + rand.nextInt(), paddedTable);  
    }
    
    public void addFeatureListener(AddFeaturesListener listener) {
        addfeaturelisteners.add(listener);
    }

    public void notifyAddFeatureListener(String name,
            ArrayList<ArrayList<Number>> feature) {
        for (AddFeaturesListener listener : addfeaturelisteners) {
            listener.addFeatures(name, "Manual classification", feature);
        }
    }

    @Override
    public void windowOpened(WindowEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void windowClosing(WindowEvent e) {
        addFeature();
    }

    @Override
    public void windowClosed(WindowEvent e) {
       
    }

    @Override
    public void windowIconified(WindowEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void windowDeiconified(WindowEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void windowActivated(WindowEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void windowDeactivated(WindowEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    interface ClassificationListener {

        public void addClassifiedObject(int classID);
    }

    class ClassLoggerWindow extends JPanel implements ActionListener {

        int objectID;

        ArrayList<ClassificationListener> ClassificationListeners = new ArrayList();

        public ClassLoggerWindow(int classes) {
            this.setLayout(new GridBagLayout());
            labelCounts = new ArrayList<Integer>();
            colorButtons = new ArrayList<JButton>();
            classDefinitions = new HashMap<>();

            // Initialize class definitions with default colors
            for (int i = 0; i < classes; i++) {
                ClassDefinition classDef = new ClassDefinition(i, "Class " + i);
                classDefinitions.put(i, classDef);
            }

            for (int i = 0; i < classes; i++) {
                // Row 0: Classification button
                GridBagConstraints gbc = new GridBagConstraints(i, 0, 1, 1, 0.2, 1.0,
                        GridBagConstraints.WEST, GridBagConstraints.BOTH, new Insets(0, 0, 0, 5), 0, 0);
                JButton button = new JButton();
                button.setName("" + i);
                button.setText("Class " + i);
                button.addActionListener(this);
                this.add(button, gbc);

                // Row 1: Class name text field
                gbc = new GridBagConstraints(i, 1, 1, 1, 0.2, 1.0,
                        GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 5), 0, 0);
                JTextField text = new JTextField(classDefinitions.get(i).getClassName());
                textFields.add(text);
                this.add(text, gbc);

                // Row 2: Color picker button
                gbc = new GridBagConstraints(i, 2, 1, 1, 0.2, 1.0,
                        GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 5), 0, 0);
                JButton colorButton = new JButton("Color");
                colorButton.setName("color_" + i);
                colorButton.setBackground(classDefinitions.get(i).getDisplayColor());
                colorButton.addActionListener(this);
                colorButtons.add(colorButton);
                this.add(colorButton, gbc);

                // Row 3: Count label
                gbc = new GridBagConstraints(i, 3, 1, 1, 0.2, 1.0,
                        GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(0, 0, 0, 5), 0, 0);
                JLabel label = new JLabel("" + 0, JLabel.CENTER);
                label.setPreferredSize(new Dimension(10, 30));
                textLabels.add(label);
                labelCounts.add(0);
                this.add(label, gbc);
            }

            // Row 4: Import/Export and Done buttons
            GridBagConstraints gbc = new GridBagConstraints(0, 4, 1, 1, 0.2, 1.0,
                    GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(5, 0, 0, 5), 0, 0);
            JButton importButton = new JButton("Import");
            importButton.setName("import");
            importButton.addActionListener(this);
            this.add(importButton, gbc);

            gbc = new GridBagConstraints(1, 4, 1, 1, 0.2, 1.0,
                    GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(5, 0, 0, 5), 0, 0);
            JButton exportButton = new JButton("Export");
            exportButton.setName("export");
            exportButton.addActionListener(this);
            this.add(exportButton, gbc);

            gbc = new GridBagConstraints(classes-1, 4, 1, 1, 0.2, 1.0,
                    GridBagConstraints.CENTER, GridBagConstraints.BOTH, new Insets(5, 0, 0, 5), 0, 0);
            JButton doneButton = new JButton("Done");
            doneButton.setName("done");
            doneButton.addActionListener(this);
            this.add(doneButton, gbc);
        }

        public void addResetSelectionListener(ClassificationListener listener) {
            ClassificationListeners.add(listener);
        }

        public void notifyResetSelectionListeners(int classID) {
            for (ClassificationListener listener : ClassificationListeners) {
                listener.addClassifiedObject(classID);
            }
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            JButton source = (JButton) e.getSource();
            String buttonName = source.getName();

            // Handle Done button
            if ("done".equals(buttonName)) {
                // Update class definitions with user-entered names
                for (int i = 0; i < textFields.size(); i++) {
                    String className = textFields.get(i).getText();
                    classDefinitions.get(i).setClassName(className);
                    classDefinitions.get(i).setSampleCount(labelCounts.get(i));
                }

                // Log classification results
                IJ.log("VTEA Manual Classification record: ");
                for (int i = 0; i < textFields.size(); i++) {
                    ClassDefinition def = classDefinitions.get(i);
                    IJ.log(String.format("Class %d: %s (samples: %d, color: %s)",
                            i, def.getClassName(), def.getSampleCount(), def.getDisplayColor()));
                }

                // Save class definitions to file (optional)
                saveClassDefinitions();

                classLoggerFrame.setVisible(false);
                classifying.close();
                addFeature();
            }
            // Handle Import button
            else if ("import".equals(buttonName)) {
                importClassDefinitions();
            }
            // Handle Export button
            else if ("export".equals(buttonName)) {
                exportClassDefinitions();
            }
            // Handle Color picker buttons
            else if (buttonName.startsWith("color_")) {
                int classId = Integer.parseInt(buttonName.substring(6));
                showColorPicker(classId);
            }
            // Handle Class assignment buttons
            else {
                try {
                    int classId = Integer.parseInt(buttonName);

                    result.put((double) currentCell, classId);

                    Integer value = labelCounts.get(classId);
                    value = value + 1;
                    labelCounts.set(classId, value);

                    JLabel label = textLabels.get(classId);
                    label.setText(value.toString());

                    classifying.close();
                    System.gc();
                    processObject();
                } catch (NumberFormatException ex) {
                    // Not a class button, ignore
                }
            }
        }

        private void showColorPicker(int classId) {
            Color currentColor = classDefinitions.get(classId).getDisplayColor();
            Color newColor = JColorChooser.showDialog(
                    classLoggerFrame,
                    "Choose Color for " + textFields.get(classId).getText(),
                    currentColor);

            if (newColor != null) {
                classDefinitions.get(classId).setDisplayColor(newColor);
                colorButtons.get(classId).setBackground(newColor);
            }
        }

        private void importClassDefinitions() {
            JFileChooser fc = new JFileChooser();
            fc.setDialogTitle("Import Class Definitions");
            fc.setFileFilter(new FileNameExtensionFilter("JSON files (*.json)", "json"));

            if (fc.showOpenDialog(classLoggerFrame) == JFileChooser.APPROVE_OPTION) {
                try {
                    File file = fc.getSelectedFile();
                    DatasetDefinition dataset = DatasetDefinition.loadFromFile(file.getAbsolutePath());

                    if (dataset.hasClassDefinitions()) {
                        classDefinitions = dataset.getClassDefinitions();

                        // Update UI with imported definitions
                        for (int i = 0; i < Math.min(textFields.size(), classDefinitions.size()); i++) {
                            ClassDefinition def = classDefinitions.get(i);
                            if (def != null) {
                                textFields.get(i).setText(def.getClassName());
                                colorButtons.get(i).setBackground(def.getDisplayColor());
                            }
                        }

                        IJ.log("Imported " + classDefinitions.size() + " class definitions from " + file.getName());
                    } else {
                        JOptionPane.showMessageDialog(classLoggerFrame,
                                "No class definitions found in file.",
                                "Import Error",
                                JOptionPane.WARNING_MESSAGE);
                    }
                } catch (IOException ex) {
                    JOptionPane.showMessageDialog(classLoggerFrame,
                            "Failed to import: " + ex.getMessage(),
                            "Import Error",
                            JOptionPane.ERROR_MESSAGE);
                }
            }
        }

        private void exportClassDefinitions() {
            // Update class definitions with current UI values
            for (int i = 0; i < textFields.size(); i++) {
                classDefinitions.get(i).setClassName(textFields.get(i).getText());
                classDefinitions.get(i).setSampleCount(labelCounts.get(i));
            }

            JFileChooser fc = new JFileChooser();
            fc.setDialogTitle("Export Class Definitions");
            fc.setFileFilter(new FileNameExtensionFilter("JSON files (*.json)", "json"));
            fc.setSelectedFile(new File("class_definitions.json"));

            if (fc.showSaveDialog(classLoggerFrame) == JFileChooser.APPROVE_OPTION) {
                try {
                    File file = fc.getSelectedFile();
                    String path = file.getAbsolutePath();
                    if (!path.endsWith(".json")) {
                        path += ".json";
                    }

                    // Create DatasetDefinition with class definitions
                    DatasetDefinition dataset = new DatasetDefinition();
                    dataset.setName("Manual Classification - " + key);
                    dataset.setClassDefinitions(classDefinitions);

                    dataset.saveToFile(path);
                    IJ.log("Exported " + classDefinitions.size() + " class definitions to " + path);
                } catch (IOException ex) {
                    JOptionPane.showMessageDialog(classLoggerFrame,
                            "Failed to export: " + ex.getMessage(),
                            "Export Error",
                            JOptionPane.ERROR_MESSAGE);
                }
            }
        }

        private void saveClassDefinitions() {
            try {
                // Auto-save to a default location
                String userHome = System.getProperty("user.home");
                String savePath = userHome + File.separator + ".vtea" + File.separator + "last_classification.json";

                File saveDir = new File(userHome + File.separator + ".vtea");
                if (!saveDir.exists()) {
                    saveDir.mkdirs();
                }

                DatasetDefinition dataset = new DatasetDefinition();
                dataset.setName("Manual Classification - " + key);
                dataset.setClassDefinitions(classDefinitions);
                dataset.saveToFile(savePath);

                IJ.log("Auto-saved class definitions to: " + savePath);
            } catch (IOException ex) {
                IJ.log("Warning: Could not auto-save class definitions: " + ex.getMessage());
            }
        }
        
        private int getTotalCount(){
            ListIterator<Integer> itr = labelCounts.listIterator();
            
            int sum = 0;
            
            while(itr.hasNext()){
                Integer n = itr.next();
                sum = sum + n;
            }
            return sum;
        }
    }

    class SetupManualClassification extends JPanel implements ItemListener {

        JSpinner nClasses;
        JSpinner nCells;
        JCheckBox allCells;

        int nCellsValue = 100;
        ArrayList<JLabel> labels = new ArrayList<JLabel>();

        public SetupManualClassification() {
            JLabel classLabel = new JLabel("Number of classes: ");
            labels.add(classLabel);
            nClasses = new JSpinner(new SpinnerNumberModel(5, 1, 20, 1));


            ListIterator<JLabel> labiter = labels.listIterator();
            setupPanel(labiter);
        }

        private void setupPanel(ListIterator<JLabel> labiter) {
            JLabel curlabel;
            this.removeAll();
            this.setLayout(new GridBagLayout());

            curlabel = labiter.next();
            GridBagConstraints gbc = new GridBagConstraints(0, 0, 1, 1, 0.2, 1.0,
                    GridBagConstraints.WEST, GridBagConstraints.BOTH, new Insets(0, 0, 0, 5), 0, 0);
            this.add(curlabel, gbc);
            gbc = new GridBagConstraints(1, 0, 1, 1, 1, 1.0, GridBagConstraints.EAST,
                    GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 5), 0, 0);
            this.add(nClasses, gbc);

        }

        public int showDialog() {
            return JOptionPane.showOptionDialog(null, this, "Setup Output Images",
                    JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE, null,
                    null, null);
        }

        public int[] getInformation() {
            int[] settings = new int[2];

            settings[0] = (int) this.nClasses.getValue();
            settings[1] = objects.size();
            
            return settings;
        }

        @Override
        public void itemStateChanged(ItemEvent e) {
            if (((JCheckBox) e.getItem()).isSelected()) {
                nCellsValue = (int) nCells.getValue();
                this.nCells.setEnabled(false);
            } else {
                nCells = new JSpinner(new SpinnerNumberModel(nCellsValue, 20, 1000, 20));
                this.nCells.setEnabled(true);
            }
        }

    }
}
