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
package vtea.deeplearning;

import ij.IJ;
import ij.Prefs;
import java.io.File;

/**
 * Configuration management for deep learning functionality.
 * Handles device selection, model paths, and default parameters.
 *
 * @author VTEA Deep Learning Team
 */
public class DeepLearningConfig {

    // Configuration keys for ImageJ preferences
    private static final String PREF_KEY_DEVICE = "vtea.deeplearning.device";
    private static final String PREF_KEY_MODEL_DIR = "vtea.deeplearning.modeldir";
    private static final String PREF_KEY_DEFAULT_BATCH_SIZE = "vtea.deeplearning.batchsize";
    private static final String PREF_KEY_DEFAULT_REGION_SIZE = "vtea.deeplearning.regionsize";
    private static final String PREF_KEY_MEMORY_LIMIT = "vtea.deeplearning.memorylimit";

    // Default values
    private static final String DEFAULT_DEVICE = "CPU";
    private static final String DEFAULT_MODEL_DIR = System.getProperty("user.home") + File.separator + ".vtea" + File.separator + "models";
    private static final int DEFAULT_BATCH_SIZE = 16;
    private static final int DEFAULT_REGION_SIZE = 64;
    private static final long DEFAULT_MEMORY_LIMIT = 8192; // 8GB in MB

    // Singleton instance
    private static DeepLearningConfig instance;

    // Instance fields
    private String device;
    private String modelDirectory;
    private int defaultBatchSize;
    private int defaultRegionSize;
    private long memoryLimitMB;
    private boolean gpuAvailable;

    /**
     * Private constructor for singleton pattern
     */
    private DeepLearningConfig() {
        loadPreferences();
        detectGPU();
    }

    /**
     * Get singleton instance
     */
    public static DeepLearningConfig getInstance() {
        if (instance == null) {
            synchronized (DeepLearningConfig.class) {
                if (instance == null) {
                    instance = new DeepLearningConfig();
                }
            }
        }
        return instance;
    }

    /**
     * Load configuration from ImageJ preferences
     */
    private void loadPreferences() {
        device = Prefs.get(PREF_KEY_DEVICE, DEFAULT_DEVICE);
        modelDirectory = Prefs.get(PREF_KEY_MODEL_DIR, DEFAULT_MODEL_DIR);
        defaultBatchSize = (int) Prefs.get(PREF_KEY_DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE);
        defaultRegionSize = (int) Prefs.get(PREF_KEY_DEFAULT_REGION_SIZE, DEFAULT_REGION_SIZE);
        memoryLimitMB = (long) Prefs.get(PREF_KEY_MEMORY_LIMIT, DEFAULT_MEMORY_LIMIT);

        // Ensure model directory exists
        File modelDir = new File(modelDirectory);
        if (!modelDir.exists()) {
            modelDir.mkdirs();
            IJ.log("Created model directory: " + modelDirectory);
        }
    }

    /**
     * Save configuration to ImageJ preferences
     */
    public void savePreferences() {
        Prefs.set(PREF_KEY_DEVICE, device);
        Prefs.set(PREF_KEY_MODEL_DIR, modelDirectory);
        Prefs.set(PREF_KEY_DEFAULT_BATCH_SIZE, defaultBatchSize);
        Prefs.set(PREF_KEY_DEFAULT_REGION_SIZE, defaultRegionSize);
        Prefs.set(PREF_KEY_MEMORY_LIMIT, memoryLimitMB);
        Prefs.savePreferences();
    }

    /**
     * Detect if GPU (CUDA) is available
     */
    private void detectGPU() {
        try {
            // Try to load CUDA classes
            Class.forName("org.bytedeco.cuda.global.cudart");
            gpuAvailable = true;
            IJ.log("GPU (CUDA) support detected");
        } catch (ClassNotFoundException e) {
            gpuAvailable = false;
            IJ.log("GPU (CUDA) not available, using CPU only");
            if ("GPU".equals(device)) {
                IJ.log("Warning: GPU selected but not available, falling back to CPU");
                device = "CPU";
            }
        }
    }

    // Getters and setters

    public String getDevice() {
        return device;
    }

    public void setDevice(String device) {
        if ("GPU".equals(device) && !gpuAvailable) {
            IJ.log("Warning: GPU requested but not available, using CPU");
            this.device = "CPU";
        } else {
            this.device = device;
        }
    }

    public boolean isGpuAvailable() {
        return gpuAvailable;
    }

    public String getModelDirectory() {
        return modelDirectory;
    }

    public void setModelDirectory(String modelDirectory) {
        this.modelDirectory = modelDirectory;
        File modelDir = new File(modelDirectory);
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
    }

    public int getDefaultBatchSize() {
        return defaultBatchSize;
    }

    public void setDefaultBatchSize(int batchSize) {
        if (batchSize < 1 || batchSize > 256) {
            throw new IllegalArgumentException("Batch size must be between 1 and 256");
        }
        this.defaultBatchSize = batchSize;
    }

    public int getDefaultRegionSize() {
        return defaultRegionSize;
    }

    public void setDefaultRegionSize(int regionSize) {
        if (regionSize < 16 || regionSize > 256 || (regionSize & (regionSize - 1)) != 0) {
            throw new IllegalArgumentException("Region size must be a power of 2 between 16 and 256");
        }
        this.defaultRegionSize = regionSize;
    }

    public long getMemoryLimitMB() {
        return memoryLimitMB;
    }

    public void setMemoryLimitMB(long memoryLimitMB) {
        if (memoryLimitMB < 1024 || memoryLimitMB > 65536) {
            throw new IllegalArgumentException("Memory limit must be between 1024 MB (1 GB) and 65536 MB (64 GB)");
        }
        this.memoryLimitMB = memoryLimitMB;
    }

    /**
     * Get full path for a model file
     */
    public String getModelPath(String modelName) {
        if (!modelName.endsWith(".pt")) {
            modelName = modelName + ".pt";
        }
        return modelDirectory + File.separator + modelName;
    }

    /**
     * Check if a model file exists
     */
    public boolean modelExists(String modelName) {
        return new File(getModelPath(modelName)).exists();
    }

    /**
     * Get available memory in MB
     */
    public long getAvailableMemoryMB() {
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long allocatedMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long availableMemory = maxMemory - allocatedMemory + freeMemory;
        return availableMemory / (1024 * 1024);
    }

    /**
     * Check if sufficient memory is available for operation
     */
    public boolean hasSufficientMemory(long requiredMB) {
        return getAvailableMemoryMB() >= requiredMB;
    }

    /**
     * Print current configuration
     */
    public void printConfig() {
        IJ.log("=== Deep Learning Configuration ===");
        IJ.log("Device: " + device);
        IJ.log("GPU Available: " + gpuAvailable);
        IJ.log("Model Directory: " + modelDirectory);
        IJ.log("Default Batch Size: " + defaultBatchSize);
        IJ.log("Default Region Size: " + defaultRegionSize);
        IJ.log("Memory Limit: " + memoryLimitMB + " MB");
        IJ.log("Available Memory: " + getAvailableMemoryMB() + " MB");
        IJ.log("===================================");
    }
}
