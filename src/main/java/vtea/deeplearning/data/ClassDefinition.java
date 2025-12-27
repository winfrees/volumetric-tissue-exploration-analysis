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
import java.awt.Color;
import java.io.Serializable;
import java.util.Date;
import java.util.Random;

/**
 * Definition of a classification class with user-provided name, color, and metadata.
 * Enables human-readable classification results instead of numeric labels.
 *
 * @author VTEA Deep Learning Team
 */
public class ClassDefinition implements Serializable {

    private static final long serialVersionUID = 1L;

    private int classId;                // Numeric identifier (0, 1, 2, ...)
    private String className;           // User-provided name (e.g., "Podocyte")
    private String description;         // Detailed description (optional)
    private int displayColorRGB;        // Stored as RGB int for serialization

    // Metadata
    private int sampleCount;            // Number of labeled samples
    private Date created;               // Creation timestamp
    private Date lastModified;          // Last update timestamp
    private String author;              // User who created this class

    /**
     * Default constructor for serialization
     */
    public ClassDefinition() {
        this.created = new Date();
        this.lastModified = new Date();
        this.sampleCount = 0;
    }

    /**
     * Constructor with ID and name
     */
    public ClassDefinition(int classId, String className) {
        this();
        this.classId = classId;
        this.className = className;
        this.displayColorRGB = generateRandomColor(classId).getRGB();
    }

    /**
     * Constructor with ID, name, and color
     */
    public ClassDefinition(int classId, String className, Color color) {
        this(classId, className);
        this.displayColorRGB = color.getRGB();
    }

    /**
     * Full constructor
     */
    public ClassDefinition(int classId, String className, String description, Color color, String author) {
        this(classId, className, color);
        this.description = description;
        this.author = author;
    }

    // Getters and setters

    public int getClassId() {
        return classId;
    }

    public void setClassId(int classId) {
        this.classId = classId;
        this.lastModified = new Date();
    }

    public String getClassName() {
        return className;
    }

    public void setClassName(String className) {
        if (className == null || className.trim().isEmpty()) {
            throw new IllegalArgumentException("Class name cannot be null or empty");
        }
        this.className = className;
        this.lastModified = new Date();
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
        this.lastModified = new Date();
    }

    public Color getDisplayColor() {
        return new Color(displayColorRGB);
    }

    public void setDisplayColor(Color color) {
        if (color == null) {
            throw new IllegalArgumentException("Color cannot be null");
        }
        this.displayColorRGB = color.getRGB();
        this.lastModified = new Date();
    }

    public int getSampleCount() {
        return sampleCount;
    }

    public void setSampleCount(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("Sample count cannot be negative");
        }
        this.sampleCount = count;
        this.lastModified = new Date();
    }

    public void incrementSampleCount() {
        this.sampleCount++;
        this.lastModified = new Date();
    }

    public void decrementSampleCount() {
        if (this.sampleCount > 0) {
            this.sampleCount--;
            this.lastModified = new Date();
        }
    }

    public Date getCreated() {
        return created;
    }

    public Date getLastModified() {
        return lastModified;
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
        this.lastModified = new Date();
    }

    // Utility methods

    /**
     * Check if class definition is valid
     */
    public boolean isValid() {
        return classId >= 0 && className != null && !className.trim().isEmpty();
    }

    /**
     * Generate a visually distinct color based on class ID
     * Uses golden ratio for better color distribution
     */
    private static Color generateRandomColor(int classId) {
        // Use golden ratio for better color distribution
        double goldenRatio = 0.618033988749895;
        double hue = (classId * goldenRatio) % 1.0;

        // Use high saturation and value for vibrant colors
        float saturation = 0.7f + (float) ((classId * 0.1) % 0.3); // 0.7-1.0
        float brightness = 0.8f + (float) ((classId * 0.05) % 0.2); // 0.8-1.0

        return Color.getHSBColor((float) hue, saturation, brightness);
    }

    /**
     * Generate a set of distinct colors for multiple classes
     */
    public static Color[] generateDistinctColors(int numClasses) {
        Color[] colors = new Color[numClasses];
        for (int i = 0; i < numClasses; i++) {
            colors[i] = generateRandomColor(i);
        }
        return colors;
    }

    // Serialization

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
    public static ClassDefinition fromJson(String json) {
        Gson gson = new Gson();
        return gson.fromJson(json, ClassDefinition.class);
    }

    @Override
    public String toString() {
        return String.format("ClassDefinition[id=%d, name=%s, color=%s, samples=%d]",
                           classId, className, getDisplayColor(), sampleCount);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;

        ClassDefinition other = (ClassDefinition) obj;
        return classId == other.classId &&
               className.equals(other.className) &&
               displayColorRGB == other.displayColorRGB;
    }

    @Override
    public int hashCode() {
        int result = classId;
        result = 31 * result + className.hashCode();
        result = 31 * result + displayColorRGB;
        return result;
    }

    /**
     * Create a copy of this class definition
     */
    public ClassDefinition copy() {
        ClassDefinition copy = new ClassDefinition(this.classId, this.className + " (copy)",
                                                    this.description, getDisplayColor(), this.author);
        copy.sampleCount = 0; // Reset sample count for copy
        return copy;
    }

    /**
     * Builder pattern for easier construction
     */
    public static class Builder {
        private ClassDefinition definition;

        public Builder(int classId, String className) {
            definition = new ClassDefinition(classId, className);
        }

        public Builder description(String description) {
            definition.description = description;
            return this;
        }

        public Builder color(Color color) {
            definition.setDisplayColor(color);
            return this;
        }

        public Builder author(String author) {
            definition.author = author;
            return this;
        }

        public Builder sampleCount(int count) {
            definition.sampleCount = count;
            return this;
        }

        public ClassDefinition build() {
            if (!definition.isValid()) {
                throw new IllegalStateException("Invalid class definition: " + definition);
            }
            return definition;
        }
    }

    /**
     * Create a default class definition with auto-generated name
     */
    public static ClassDefinition createDefault(int classId) {
        return new ClassDefinition(classId, "Class_" + classId);
    }

    /**
     * Create default class definitions for a given number of classes
     */
    public static ClassDefinition[] createDefaults(int numClasses) {
        ClassDefinition[] definitions = new ClassDefinition[numClasses];
        for (int i = 0; i < numClasses; i++) {
            definitions[i] = createDefault(i);
        }
        return definitions;
    }
}
