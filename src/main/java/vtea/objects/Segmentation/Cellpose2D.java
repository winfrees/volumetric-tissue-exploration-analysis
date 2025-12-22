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
package vtea.objects.Segmentation;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.process.ImageProcessor;
import java.awt.Dimension;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import org.scijava.plugin.Plugin;
import vtea.objects.layercake.microVolume;
import vteaobjects.MicroObject;

/**
 * Cellpose segmentation for 2D images.
 *
 * @author Claude/Seth Winfree
 */
@Plugin(type = Segmentation.class)
public class Cellpose2D extends AbstractSegmentation {

    private ImagePlus imageOriginal;
    private ImagePlus imageResult;
    private ImageStack stackOriginal;
    protected ImageStack stackResult;

    private ArrayList<MicroObject> alVolumes = new ArrayList<MicroObject>();

    JTextField diameterField = new JTextField("30", 5);
    JComboBox<String> modelCombo = new JComboBox<>(new String[]{"cyto", "nuclei", "cyto2"});
    JTextField flowThresholdField = new JTextField("0.4", 5);
    JTextField cellprobThresholdField = new JTextField("0.0", 5);

    public Cellpose2D() {
        VERSION = "0.1";
        AUTHOR = "Claude/Seth Winfree";
        COMMENT = "Cellpose deep learning segmentation for 2D images.";
        NAME = "Cellpose 2D";
        KEY = "Cellpose2D";
        COMPATIBILITY = "2D";
        TYPE = "Calculated";

        protocol = new ArrayList();

        diameterField.setPreferredSize(new Dimension(50, 30));
        diameterField.setMaximumSize(diameterField.getPreferredSize());
        diameterField.setMinimumSize(diameterField.getPreferredSize());

        flowThresholdField.setPreferredSize(new Dimension(50, 30));
        flowThresholdField.setMaximumSize(flowThresholdField.getPreferredSize());
        flowThresholdField.setMinimumSize(flowThresholdField.getPreferredSize());

        cellprobThresholdField.setPreferredSize(new Dimension(50, 30));
        cellprobThresholdField.setMaximumSize(cellprobThresholdField.getPreferredSize());
        cellprobThresholdField.setMinimumSize(cellprobThresholdField.getPreferredSize());

        protocol.add(new JLabel("Diameter (pixels)"));
        protocol.add(diameterField);
        protocol.add(new JLabel("Model"));
        protocol.add(modelCombo);
        protocol.add(new JLabel("Flow Threshold"));
        protocol.add(flowThresholdField);
        protocol.add(new JLabel("Cellprob Threshold"));
        protocol.add(cellprobThresholdField);
    }

    @Override
    public void setImage(ImagePlus thresholdPreview) {
        imagePreview = thresholdPreview;
    }

    @Override
    public void updateImage(ImagePlus thresholdPreview) {
        imagePreview = thresholdPreview;
    }

    @Override
    public ArrayList<MicroObject> getObjects() {
        return alVolumes;
    }

    @Override
    public ImagePlus getSegmentation() {
        return this.imageResult;
    }

    @Override
    public JPanel getSegmentationTool() {
        JPanel panel = new JPanel();
        panel.setBackground(vtea._vtea.BACKGROUND);
        return panel;
    }

    @Override
    public void doUpdateOfTool() {
        // No dynamic updates needed for Cellpose
    }

    @Override
    public boolean copyComponentParameter(String version, ArrayList dComponents, ArrayList sComponents) {
        try {
            dComponents.clear();

            JTextField diameter = (JTextField) sComponents.get(1);
            JComboBox<String> model = (JComboBox<String>) sComponents.get(3);
            JTextField flowThreshold = (JTextField) sComponents.get(5);
            JTextField cellprobThreshold = (JTextField) sComponents.get(7);

            dComponents.add(new JLabel("Diameter (pixels)"));
            dComponents.add(diameter);
            dComponents.add(new JLabel("Model"));
            dComponents.add(model);
            dComponents.add(new JLabel("Flow Threshold"));
            dComponents.add(flowThreshold);
            dComponents.add(new JLabel("Cellprob Threshold"));
            dComponents.add(cellprobThreshold);

            return true;
        } catch (Exception e) {
            System.out.println("ERROR: Could not copy parameter(s) for " + NAME);
            return false;
        }
    }

    @Override
    public boolean loadComponentParameter(String version, ArrayList dComponents, ArrayList fields) {
        try {
            JTextField diameter = (JTextField) dComponents.get(1);
            JComboBox<String> model = (JComboBox<String>) dComponents.get(3);
            JTextField flowThreshold = (JTextField) dComponents.get(5);
            JTextField cellprobThreshold = (JTextField) dComponents.get(7);

            diameter.setText((String) fields.get(0));
            model.setSelectedItem((String) fields.get(1));
            flowThreshold.setText((String) fields.get(2));
            cellprobThreshold.setText((String) fields.get(3));

            return true;
        } catch (Exception e) {
            System.out.println("ERROR: Could not load parameter(s) for " + NAME);
            return false;
        }
    }

    @Override
    public boolean saveComponentParameter(String version, ArrayList fields, ArrayList sComponents) {
        try {
            JTextField diameter = (JTextField) sComponents.get(1);
            JComboBox<String> model = (JComboBox<String>) sComponents.get(3);
            JTextField flowThreshold = (JTextField) sComponents.get(5);
            JTextField cellprobThreshold = (JTextField) sComponents.get(7);

            fields.add(diameter.getText());
            fields.add((String) model.getSelectedItem());
            fields.add(flowThreshold.getText());
            fields.add(cellprobThreshold.getText());

            return true;
        } catch (Exception e) {
            System.out.println("ERROR: Could not save parameter(s) for " + NAME + "\n" + e.getLocalizedMessage());
            return false;
        }
    }

    @Override
    public boolean process(ImageStack[] is, List protocol, boolean calculate) {
        System.out.println("PROFILING: processing on Cellpose 2D...");

        ArrayList al = (ArrayList) protocol.get(3);

        int diameter = Integer.parseInt(((JTextField) (al.get(1))).getText());
        String model = (String) ((JComboBox) (al.get(3))).getSelectedItem();
        double flowThreshold = Double.parseDouble(((JTextField) (al.get(5))).getText());
        double cellprobThreshold = Double.parseDouble(((JTextField) (al.get(7))).getText());

        int segmentationChannel = (int) protocol.get(2);

        stackOriginal = is[segmentationChannel];
        imageOriginal = new ImagePlus("Original", stackOriginal);

        // For 2D, we only process the first slice
        ImageProcessor ip = stackOriginal.getProcessor(1);
        ImagePlus singleSlice = new ImagePlus("Slice", ip);

        try {
            // Save image to temporary file
            String tempDir = System.getProperty("java.io.tmpdir");
            String inputPath = tempDir + File.separator + "cellpose_input_" + System.currentTimeMillis() + ".tif";
            String outputPath = tempDir + File.separator + "cellpose_output_" + System.currentTimeMillis() + ".tif";

            FileSaver fs = new FileSaver(singleSlice);
            fs.saveAsTiff(inputPath);

            notifyProgressListeners("Running Cellpose...", 25.0);

            // Execute Cellpose Python script
            String pythonPath = getPython();
            String scriptPath = getCWD() + String.format("%1$csrc%1$cmain%1$cresources%1$ccellpose_segment.py", File.separatorChar);

            String[] command = new String[]{
                pythonPath,
                scriptPath,
                inputPath,
                outputPath,
                String.valueOf(diameter),
                model,
                String.valueOf(flowThreshold),
                String.valueOf(cellprobThreshold)
            };

            Process p = Runtime.getRuntime().exec(command);

            // Read stdout and stderr
            BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));

            String s;
            while ((s = stdInput.readLine()) != null) {
                System.out.println(s);
            }

            String e;
            while ((e = stdError.readLine()) != null) {
                System.err.println(e);
            }

            int exitCode = p.waitFor();

            if (exitCode != 0) {
                System.err.println("ERROR: Cellpose script failed with exit code " + exitCode);
                return false;
            }

            notifyProgressListeners("Processing results...", 75.0);

            // Load result image
            imageResult = IJ.openImage(outputPath);

            if (imageResult == null) {
                System.err.println("ERROR: Could not load Cellpose output");
                return false;
            }

            stackResult = imageResult.getStack();

            // Convert labels to MicroObjects
            convertLabelsToObjects(stackResult, stackOriginal);

            // Clean up temporary files
            new File(inputPath).delete();
            new File(outputPath).delete();

            notifyProgressListeners("Complete", 100.0);

            System.out.println("PROFILING: Found " + alVolumes.size() + " objects.");

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return false;
        }

        return true;
    }

    /**
     * Convert label image to MicroObject list
     */
    private void convertLabelsToObjects(ImageStack labelStack, ImageStack originalStack) {
        ImageProcessor labelProc = labelStack.getProcessor(1);

        // Find all unique labels
        HashMap<Integer, ArrayList<int[]>> labelPixels = new HashMap<>();

        for (int x = 0; x < labelProc.getWidth(); x++) {
            for (int y = 0; y < labelProc.getHeight(); y++) {
                int label = (int) labelProc.getPixelValue(x, y);

                if (label > 0) {  // 0 is background
                    if (!labelPixels.containsKey(label)) {
                        labelPixels.put(label, new ArrayList<>());
                    }
                    labelPixels.get(label).add(new int[]{x, y});
                }
            }
        }

        // Convert each label to a MicroObject
        for (Integer label : labelPixels.keySet()) {
            ArrayList<int[]> pixels = labelPixels.get(label);

            int[] xPixels = new int[pixels.size()];
            int[] yPixels = new int[pixels.size()];
            int[] zPixels = new int[pixels.size()];

            for (int i = 0; i < pixels.size(); i++) {
                xPixels[i] = pixels.get(i)[0];
                yPixels[i] = pixels.get(i)[1];
                zPixels[i] = 0;  // 2D, so z=0
            }

            microVolume mv = new microVolume();
            mv.setPixelsX(xPixels);
            mv.setPixelsY(yPixels);
            mv.setPixelsZ(zPixels);
            mv.setCentroid();
            mv.setSerialID(alVolumes.size());

            alVolumes.add(mv);
        }
    }

    /**
     * Helper method to get Python executable path
     */
    private String getPython() {
        Process p;
        String s = "";
        try {
            if (isWindows()) {
                p = Runtime.getRuntime().exec("cmd.exe /c where python");
            } else {
                p = Runtime.getRuntime().exec("which python");
            }
            p.waitFor();
            BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
            s = stdInput.readLine();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return s != null ? s : "python";
    }

    /**
     * Helper method to get current working directory
     */
    private String getCWD() {
        return Paths.get("").toAbsolutePath().toString();
    }

    /**
     * Helper method to check if running on Windows
     */
    private boolean isWindows() {
        return System.getProperty("os.name").toLowerCase().startsWith("windows");
    }
}
