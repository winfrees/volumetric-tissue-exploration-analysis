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
package vtea.deeplearning.data;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.*;
import java.util.Date;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * Configuration class for dataset definition.
 * Defines how to extract and preprocess regions for deep learning.
 * Includes class definitions for user-defined class naming.
 *
 * @author VTEA Deep Learning Team
 */
public class DatasetDefinition implements Serializable {

    private static final long serialVersionUID = 1L;

    // Region extraction parameters
    private int[] regionSize;           // [depth, height, width]
    private int[] channels;             // channel indices to extract
    private CellRegionExtractor.PaddingType paddingType;
    private int padding;                // padding amount for boundaries

    // Preprocessing parameters
    private boolean normalize;          // apply normalization
    private TensorConverter.NormalizationType normType;

    // Class definitions (Phase 3)
    private HashMap<Integer, ClassDefinition> classDefinitions;

    // Metadata
    private String name;
    private String description;
    private Date created;
    private Date lastModified;

    /**
     * Default constructor with reasonable defaults
     */
    public DatasetDefinition() {
        this.regionSize = new int[]{64, 64, 64};
        this.channels = new int[]{0};  // First channel only
        this.paddingType = CellRegionExtractor.PaddingType.ZERO;
        this.padding = 0;
        this.normalize = true;
        this.normType = TensorConverter.NormalizationType.ZSCORE;
        this.classDefinitions = new HashMap<>();
        this.name = "Unnamed Dataset";
        this.description = "";
        this.created = new Date();
        this.lastModified = new Date();
    }

    /**
     * Constructor with basic parameters
     */
    public DatasetDefinition(String name, int[] regionSize, int[] channels) {
        this();
        this.name = name;
        this.regionSize = regionSize;
        this.channels = channels;
    }

    // Getters and setters

    public int[] getRegionSize() {
        return regionSize;
    }

    public void setRegionSize(int[] regionSize) {
        if (regionSize == null || regionSize.length != 3) {
            throw new IllegalArgumentException("Region size must be [depth, height, width]");
        }
        for (int size : regionSize) {
            if (size < 8 || size > 512) {
                throw new IllegalArgumentException("Region size dimensions must be between 8 and 512");
            }
        }
        this.regionSize = regionSize;
        this.lastModified = new Date();
    }

    public int[] getChannels() {
        return channels;
    }

    public void setChannels(int[] channels) {
        if (channels == null || channels.length == 0) {
            throw new IllegalArgumentException("Channels array cannot be null or empty");
        }
        this.channels = channels;
        this.lastModified = new Date();
    }

    public CellRegionExtractor.PaddingType getPaddingType() {
        return paddingType;
    }

    public void setPaddingType(CellRegionExtractor.PaddingType paddingType) {
        this.paddingType = paddingType;
        this.lastModified = new Date();
    }

    public int getPadding() {
        return padding;
    }

    public void setPadding(int padding) {
        if (padding < 0) {
            throw new IllegalArgumentException("Padding cannot be negative");
        }
        this.padding = padding;
        this.lastModified = new Date();
    }

    public boolean isNormalize() {
        return normalize;
    }

    public void setNormalize(boolean normalize) {
        this.normalize = normalize;
        this.lastModified = new Date();
    }

    public TensorConverter.NormalizationType getNormType() {
        return normType;
    }

    public void setNormType(TensorConverter.NormalizationType normType) {
        this.normType = normType;
        this.lastModified = new Date();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
        this.lastModified = new Date();
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
        this.lastModified = new Date();
    }

    public Date getCreated() {
        return created;
    }

    public Date getLastModified() {
        return lastModified;
    }

    // Class definition management methods

    /**
     * Add a class definition
     */
    public void addClassDefinition(ClassDefinition classDef) {
        if (classDef == null) {
            throw new IllegalArgumentException("ClassDefinition cannot be null");
        }
        classDefinitions.put(classDef.getClassId(), classDef);
        this.lastModified = new Date();
    }

    /**
     * Remove a class definition
     */
    public void removeClassDefinition(int classId) {
        classDefinitions.remove(classId);
        this.lastModified = new Date();
    }

    /**
     * Get a class definition by ID
     */
    public ClassDefinition getClassDefinition(int classId) {
        return classDefinitions.get(classId);
    }

    /**
     * Get all class definitions
     */
    public HashMap<Integer, ClassDefinition> getClassDefinitions() {
        return classDefinitions;
    }

    /**
     * Set all class definitions
     */
    public void setClassDefinitions(HashMap<Integer, ClassDefinition> classDefinitions) {
        this.classDefinitions = classDefinitions != null ? classDefinitions : new HashMap<>();
        this.lastModified = new Date();
    }

    /**
     * Get class name by ID
     */
    public String getClassName(int classId) {
        ClassDefinition def = classDefinitions.get(classId);
        return def != null ? def.getClassName() : "Unknown_" + classId;
    }

    /**
     * Set class name for a given ID
     */
    public void setClassName(int classId, String name) {
        ClassDefinition def = classDefinitions.get(classId);
        if (def != null) {
            def.setClassName(name);
        } else {
            classDefinitions.put(classId, new ClassDefinition(classId, name));
        }
        this.lastModified = new Date();
    }

    /**
     * Get all class names as a list
     */
    public List<String> getAllClassNames() {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < classDefinitions.size(); i++) {
            names.add(getClassName(i));
        }
        return names;
    }

    /**
     * Get number of defined classes
     */
    public int getNumClasses() {
        return classDefinitions.size();
    }

    /**
     * Check if class definitions are present
     */
    public boolean hasClassDefinitions() {
        return !classDefinitions.isEmpty();
    }

    /**
     * Initialize default class definitions for a given number of classes
     */
    public void initializeDefaultClassDefinitions(int numClasses) {
        classDefinitions.clear();
        for (int i = 0; i < numClasses; i++) {
            classDefinitions.put(i, ClassDefinition.createDefault(i));
        }
        this.lastModified = new Date();
    }

    // Utility methods

    /**
     * Get number of channels
     */
    public int getNumChannels() {
        return channels.length;
    }

    /**
     * Get total number of voxels in a region
     */
    public int getRegionVoxelCount() {
        return regionSize[0] * regionSize[1] * regionSize[2];
    }

    /**
     * Get region depth
     */
    public int getRegionDepth() {
        return regionSize[0];
    }

    /**
     * Get region height
     */
    public int getRegionHeight() {
        return regionSize[1];
    }

    /**
     * Get region width
     */
    public int getRegionWidth() {
        return regionSize[2];
    }

    /**
     * Check if region is cubic (all dimensions equal)
     */
    public boolean isRegionCubic() {
        return regionSize[0] == regionSize[1] && regionSize[1] == regionSize[2];
    }

    /**
     * Validate dataset definition
     */
    public boolean isValid() {
        if (regionSize == null || regionSize.length != 3) {
            return false;
        }
        if (channels == null || channels.length == 0) {
            return false;
        }
        if (paddingType == null || normType == null) {
            return false;
        }
        return true;
    }

    // Serialization methods

    /**
     * Save dataset definition to JSON file
     */
    public void saveToFile(String filepath) throws IOException {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try (Writer writer = new FileWriter(filepath)) {
            gson.toJson(this, writer);
        }
    }

    /**
     * Load dataset definition from JSON file
     */
    public static DatasetDefinition loadFromFile(String filepath) throws IOException {
        Gson gson = new Gson();
        try (Reader reader = new FileReader(filepath)) {
            return gson.fromJson(reader, DatasetDefinition.class);
        }
    }

    /**
     * Convert to JSON string
     */
    public String toJson() {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(this);
    }

    /**
     * Create from JSON string
     */
    public static DatasetDefinition fromJson(String json) {
        Gson gson = new Gson();
        return gson.fromJson(json, DatasetDefinition.class);
    }

    @Override
    public String toString() {
        return String.format("DatasetDefinition[name=%s, regionSize=%s, channels=%s, " +
                           "paddingType=%s, normType=%s, normalize=%b]",
                           name, Arrays.toString(regionSize), Arrays.toString(channels),
                           paddingType, normType, normalize);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;

        DatasetDefinition other = (DatasetDefinition) obj;
        return Arrays.equals(regionSize, other.regionSize) &&
               Arrays.equals(channels, other.channels) &&
               paddingType == other.paddingType &&
               padding == other.padding &&
               normalize == other.normalize &&
               normType == other.normType;
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(regionSize);
        result = 31 * result + Arrays.hashCode(channels);
        result = 31 * result + (paddingType != null ? paddingType.hashCode() : 0);
        result = 31 * result + padding;
        result = 31 * result + (normalize ? 1 : 0);
        result = 31 * result + (normType != null ? normType.hashCode() : 0);
        return result;
    }

    /**
     * Create a copy of this dataset definition
     */
    public DatasetDefinition copy() {
        DatasetDefinition copy = new DatasetDefinition();
        copy.regionSize = this.regionSize.clone();
        copy.channels = this.channels.clone();
        copy.paddingType = this.paddingType;
        copy.padding = this.padding;
        copy.normalize = this.normalize;
        copy.normType = this.normType;

        // Deep copy class definitions
        copy.classDefinitions = new HashMap<>();
        for (Integer key : this.classDefinitions.keySet()) {
            copy.classDefinitions.put(key, this.classDefinitions.get(key).copy());
        }

        copy.name = this.name + " (copy)";
        copy.description = this.description;
        copy.created = new Date();
        copy.lastModified = new Date();
        return copy;
    }

    /**
     * Builder pattern for easier construction
     */
    public static class Builder {
        private DatasetDefinition definition;

        public Builder() {
            definition = new DatasetDefinition();
        }

        public Builder name(String name) {
            definition.name = name;
            return this;
        }

        public Builder description(String description) {
            definition.description = description;
            return this;
        }

        public Builder regionSize(int depth, int height, int width) {
            definition.regionSize = new int[]{depth, height, width};
            return this;
        }

        public Builder regionSize(int size) {
            definition.regionSize = new int[]{size, size, size};
            return this;
        }

        public Builder channels(int... channels) {
            definition.channels = channels;
            return this;
        }

        public Builder paddingType(CellRegionExtractor.PaddingType paddingType) {
            definition.paddingType = paddingType;
            return this;
        }

        public Builder padding(int padding) {
            definition.padding = padding;
            return this;
        }

        public Builder normalize(boolean normalize) {
            definition.normalize = normalize;
            return this;
        }

        public Builder normalizationType(TensorConverter.NormalizationType normType) {
            definition.normType = normType;
            return this;
        }

        public DatasetDefinition build() {
            if (!definition.isValid()) {
                throw new IllegalStateException("Invalid dataset definition");
            }
            return definition;
        }
    }
}
