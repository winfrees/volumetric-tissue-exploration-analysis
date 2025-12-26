/*
 * Copyright (C) 2020 Indiana University
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
package vteaobjects;

import ij.ImageStack;
import ij.process.ByteProcessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Rectangle;
import java.util.ArrayList;

import static org.assertj.core.api.Assertions.*;

/**
 * Test class for MicroObject
 *
 * MicroObject is the core data model class that represents a segmented object
 * in 3D space, storing pixel coordinates, morphological associations, and
 * various calculated properties.
 */
@DisplayName("MicroObject Tests")
class MicroObjectTest {

    private MicroObject microObject;
    private MicroObject microObjectWithParams;
    private ArrayList<int[]> testPixels;
    private ImageStack[] testImageStacks;

    private static final double DELTA = 0.001;

    @BeforeEach
    void setUp() {
        // Create default MicroObject
        microObject = new MicroObject();

        // Create test pixels (a small 3D cube)
        testPixels = createTestPixels();

        // Create test image stacks
        testImageStacks = createTestImageStacks(100, 100, 10);

        // Create parameterized MicroObject
        microObjectWithParams = new MicroObject(testPixels, 0, testImageStacks, 42);
    }

    // ========== Constructor Tests ==========

    @Test
    @DisplayName("Default constructor should create valid instance")
    void testDefaultConstructor() {
        assertThat(microObject).isNotNull();
        assertThat(microObject.getPixelsX()).isNull();
        assertThat(microObject.getPixelsY()).isNull();
        assertThat(microObject.getPixelsZ()).isNull();
    }

    @Test
    @DisplayName("Parameterized constructor should store pixels")
    void testParameterizedConstructorStoresPixels() {
        int[] xPixels = microObjectWithParams.getPixelsX();
        int[] yPixels = microObjectWithParams.getPixelsY();
        int[] zPixels = microObjectWithParams.getPixelsZ();

        assertThat(xPixels).isNotNull();
        assertThat(yPixels).isNotNull();
        assertThat(zPixels).isNotNull();

        assertThat(xPixels).isEqualTo(testPixels.get(0));
        assertThat(yPixels).isEqualTo(testPixels.get(1));
        assertThat(zPixels).isEqualTo(testPixels.get(2));
    }

    @Test
    @DisplayName("Parameterized constructor should set serial ID")
    void testParameterizedConstructorSetsSerialID() {
        assertThat(microObjectWithParams.getSerialID()).isEqualTo(42.0);
    }

    @Test
    @DisplayName("Parameterized constructor should calculate centroid")
    void testParameterizedConstructorCalculatesCentroid() {
        // Test pixels range from 10-12 in X, 20-22 in Y, 5-7 in Z
        // Centroids should be at midpoints: 11, 21, 6
        assertThat(microObjectWithParams.getCentroidX()).isEqualTo(11.0f, within(0.01f));
        assertThat(microObjectWithParams.getCentroidY()).isEqualTo(21.0f, within(0.01f));
        assertThat(microObjectWithParams.getCentroidZ()).isEqualTo(6.0f, within(0.01f));
    }

    @Test
    @DisplayName("Parameterized constructor should set min and max Z")
    void testParameterizedConstructorSetsMinMaxZ() {
        assertThat(microObjectWithParams.getMinZ()).isEqualTo(5);
        assertThat(microObjectWithParams.getMaxZ()).isEqualTo(7);
    }

    // ========== Pixel Getter Tests ==========

    @Test
    @DisplayName("getPixelsX() should return X coordinates")
    void testGetPixelsX() {
        int[] expected = {10, 11, 12, 10, 11, 12, 10, 11, 12};
        int[] actual = microObjectWithParams.getPixelsX();

        assertThat(actual).isEqualTo(expected);
    }

    @Test
    @DisplayName("getPixelsY() should return Y coordinates")
    void testGetPixelsY() {
        int[] expected = {20, 20, 20, 21, 21, 21, 22, 22, 22};
        int[] actual = microObjectWithParams.getPixelsY();

        assertThat(actual).isEqualTo(expected);
    }

    @Test
    @DisplayName("getPixelsZ() should return Z coordinates")
    void testGetPixelsZ() {
        int[] expected = {5, 5, 5, 6, 6, 6, 7, 7, 7};
        int[] actual = microObjectWithParams.getPixelsZ();

        assertThat(actual).isEqualTo(expected);
    }

    // ========== Pixel Setter Tests ==========

    @Test
    @DisplayName("setPixelsX() should update X coordinates")
    void testSetPixelsX() {
        int[] newX = {1, 2, 3};
        microObject.setPixelsX(newX);

        assertThat(microObject.getPixelsX()).isEqualTo(newX);
    }

    @Test
    @DisplayName("setPixelsY() should update Y coordinates")
    void testSetPixelsY() {
        int[] newY = {4, 5, 6};
        microObject.setPixelsY(newY);

        assertThat(microObject.getPixelsY()).isEqualTo(newY);
    }

    @Test
    @DisplayName("setPixelsZ() should update Z coordinates")
    void testSetPixelsZ() {
        int[] newZ = {7, 8, 9};
        microObject.setPixelsZ(newZ);

        assertThat(microObject.getPixelsZ()).isEqualTo(newZ);
    }

    @Test
    @DisplayName("setPixelsX() should create new array")
    void testSetPixelsXCreatesNewArray() {
        int[] original = {1, 2, 3};
        microObject.setPixelsX(original);
        original[0] = 999;  // Modify original

        assertThat(microObject.getPixelsX()[0]).isEqualTo(1);  // Should not be affected
    }

    // ========== Centroid Tests ==========

    @Test
    @DisplayName("getCentroidX() should return X centroid")
    void testGetCentroidX() {
        float centroidX = microObjectWithParams.getCentroidX();
        assertThat(centroidX).isEqualTo(11.0f, within(0.01f));
    }

    @Test
    @DisplayName("getCentroidY() should return Y centroid")
    void testGetCentroidY() {
        float centroidY = microObjectWithParams.getCentroidY();
        assertThat(centroidY).isEqualTo(21.0f, within(0.01f));
    }

    @Test
    @DisplayName("getCentroidZ() should return Z centroid")
    void testGetCentroidZ() {
        float centroidZ = microObjectWithParams.getCentroidZ();
        assertThat(centroidZ).isEqualTo(6.0f, within(0.01f));
    }

    @Test
    @DisplayName("setCentroid() should recalculate centroid after pixel changes")
    void testSetCentroidRecalculates() {
        // Change pixels
        microObjectWithParams.setPixelsX(new int[]{0, 10});
        microObjectWithParams.setPixelsY(new int[]{0, 20});
        microObjectWithParams.setPixelsZ(new int[]{0, 10});

        // Recalculate
        microObjectWithParams.setCentroid();

        // New centroid should be at midpoint: (5, 10, 5)
        assertThat(microObjectWithParams.getCentroidX()).isEqualTo(5.0f, within(0.01f));
        assertThat(microObjectWithParams.getCentroidY()).isEqualTo(10.0f, within(0.01f));
        assertThat(microObjectWithParams.getCentroidZ()).isEqualTo(5.0f, within(0.01f));
    }

    @Test
    @DisplayName("Centroid should handle single pixel")
    void testCentroidSinglePixel() {
        ArrayList<int[]> singlePixel = new ArrayList<>();
        singlePixel.add(new int[]{42});
        singlePixel.add(new int[]{84});
        singlePixel.add(new int[]{10});

        MicroObject single = new MicroObject(singlePixel, 0, testImageStacks, 1);

        assertThat(single.getCentroidX()).isEqualTo(42.0f, within(0.01f));
        assertThat(single.getCentroidY()).isEqualTo(84.0f, within(0.01f));
        assertThat(single.getCentroidZ()).isEqualTo(10.0f, within(0.01f));
    }

    // ========== Min/Max Z Tests ==========

    @Test
    @DisplayName("getMinZ() should return minimum Z value")
    void testGetMinZ() {
        assertThat(microObjectWithParams.getMinZ()).isEqualTo(5);
    }

    @Test
    @DisplayName("getMaxZ() should return maximum Z value")
    void testGetMaxZ() {
        assertThat(microObjectWithParams.getMaxZ()).isEqualTo(7);
    }

    @Test
    @DisplayName("setMinZ() should calculate and return minimum Z")
    void testSetMinZ() {
        microObjectWithParams.setPixelsZ(new int[]{3, 1, 5, 2, 4});
        int min = microObjectWithParams.setMinZ();

        assertThat(min).isEqualTo(1);
        assertThat(microObjectWithParams.getMinZ()).isEqualTo(1);
    }

    @Test
    @DisplayName("setMaxZ() should calculate and return maximum Z")
    void testSetMaxZ() {
        microObjectWithParams.setPixelsZ(new int[]{3, 10, 5, 2, 4});
        int max = microObjectWithParams.setMaxZ();

        assertThat(max).isEqualTo(10);
        assertThat(microObjectWithParams.getMaxZ()).isEqualTo(10);
    }

    // ========== Range Tests ==========

    @Test
    @DisplayName("getRange(0) should return X range")
    void testGetRangeX() {
        // X coordinates are 10, 11, 12 -> range = 2
        int range = microObjectWithParams.getRange(0);
        assertThat(range).isEqualTo(2);
    }

    @Test
    @DisplayName("getRange(1) should return Y range")
    void testGetRangeY() {
        // Y coordinates are 20, 21, 22 -> range = 2
        int range = microObjectWithParams.getRange(1);
        assertThat(range).isEqualTo(2);
    }

    @Test
    @DisplayName("getRange(2) should return Z range")
    void testGetRangeZ() {
        // Z coordinates are 5, 6, 7 -> range = 2
        int range = microObjectWithParams.getRange(2);
        assertThat(range).isEqualTo(2);
    }

    @Test
    @DisplayName("getRange() should handle single value")
    void testGetRangeSingleValue() {
        microObjectWithParams.setPixelsX(new int[]{5, 5, 5});
        int range = microObjectWithParams.getRange(0);
        assertThat(range).isEqualTo(0);
    }

    // ========== Serial ID Tests ==========

    @Test
    @DisplayName("getSerialID() should return serial ID")
    void testGetSerialID() {
        assertThat(microObjectWithParams.getSerialID()).isEqualTo(42.0);
    }

    @Test
    @DisplayName("setSerialID() should update serial ID")
    void testSetSerialID() {
        microObject.setSerialID(123);
        assertThat(microObject.getSerialID()).isEqualTo(123.0);
    }

    // ========== Gating Tests ==========

    @Test
    @DisplayName("Default gated status should be false")
    void testDefaultGatedStatus() {
        assertThat(microObject.getGated()).isFalse();
    }

    @Test
    @DisplayName("setGated(true) should set gated to true")
    void testSetGatedTrue() {
        microObject.setGated(true);
        assertThat(microObject.getGated()).isTrue();
    }

    @Test
    @DisplayName("setGated(false) should set gated to false")
    void testSetGatedFalse() {
        microObject.setGated(true);
        microObject.setGated(false);
        assertThat(microObject.getGated()).isFalse();
    }

    // ========== Color Tests ==========

    @Test
    @DisplayName("Default color should be 0")
    void testDefaultColor() {
        assertThat(microObject.getColor()).isEqualTo(0);
    }

    @Test
    @DisplayName("setColor() should update color")
    void testSetColor() {
        microObject.setColor(5);
        assertThat(microObject.getColor()).isEqualTo(5);
    }

    @Test
    @DisplayName("Color should handle negative values")
    void testSetColorNegative() {
        microObject.setColor(-1);
        assertThat(microObject.getColor()).isEqualTo(-1);
    }

    // ========== Morphological Tests ==========

    @Test
    @DisplayName("getMorphologicalCount() should return 0 initially")
    void testGetMorphologicalCountInitially() {
        assertThat(microObject.getMorphologicalCount()).isEqualTo(0);
    }

    @Test
    @DisplayName("setMorphological() should add morphological data")
    void testSetMorphological() {
        int[] morphX = {1, 2, 3};
        int[] morphY = {4, 5, 6};
        int[] morphZ = {7, 8, 9};

        microObject.setMorphological("morph1", morphX, morphY, morphZ);

        assertThat(microObject.getMorphologicalCount()).isEqualTo(1);
    }

    @Test
    @DisplayName("setMorphological() should store multiple morphologies")
    void testSetMultipleMorphological() {
        microObject.setMorphological("morph1", new int[]{1}, new int[]{2}, new int[]{3});
        microObject.setMorphological("morph2", new int[]{4}, new int[]{5}, new int[]{6});
        microObject.setMorphological("morph3", new int[]{7}, new int[]{8}, new int[]{9});

        assertThat(microObject.getMorphologicalCount()).isEqualTo(3);
    }

    @Test
    @DisplayName("checkMorphological() should return -1 for unknown UID")
    void testCheckMorphologicalUnknown() {
        int result = microObject.checkMorphological("unknown");
        assertThat(result).isEqualTo(-1);
    }

    @Test
    @DisplayName("checkMorphological() should return index for known UID")
    void testCheckMorphologicalKnown() {
        microObject.setMorphological("morph1", new int[]{1}, new int[]{2}, new int[]{3});
        microObject.setMorphological("morph2", new int[]{4}, new int[]{5}, new int[]{6});

        int index = microObject.checkMorphological("morph2");
        assertThat(index).isEqualTo(1);
    }

    @Test
    @DisplayName("getMorphological() should return morphological data")
    void testGetMorphological() {
        int[] morphX = {1, 2, 3};
        int[] morphY = {4, 5, 6};
        int[] morphZ = {7, 8, 9};

        microObject.setMorphological("morph1", morphX, morphY, morphZ);

        ArrayList<int[]> result = microObject.getMorphological(0);

        assertThat(result).hasSize(3);
        assertThat(result.get(0)).isEqualTo(morphX);
        assertThat(result.get(1)).isEqualTo(morphY);
        assertThat(result.get(2)).isEqualTo(morphZ);
    }

    @Test
    @DisplayName("getMorphPixelsX() should return X pixels for morphology")
    void testGetMorphPixelsX() {
        int[] morphX = {10, 20, 30};
        microObject.setMorphological("morph1", morphX, new int[]{1}, new int[]{1});

        int[] result = microObject.getMorphPixelsX(0);
        assertThat(result).isEqualTo(morphX);
    }

    @Test
    @DisplayName("getMorphPixelsY() should return Y pixels for morphology")
    void testGetMorphPixelsY() {
        int[] morphY = {40, 50, 60};
        microObject.setMorphological("morph1", new int[]{1}, morphY, new int[]{1});

        int[] result = microObject.getMorphPixelsY(0);
        assertThat(result).isEqualTo(morphY);
    }

    @Test
    @DisplayName("getMorphPixelsZ() should return Z pixels for morphology")
    void testGetMorphPixelsZ() {
        int[] morphZ = {70, 80, 90};
        microObject.setMorphological("morph1", new int[]{1}, new int[]{1}, morphZ);

        int[] result = microObject.getMorphPixelsZ(0);
        assertThat(result).isEqualTo(morphZ);
    }

    // ========== Region Tests ==========

    @Test
    @DisplayName("getXPixelsInRegion() should return X pixels in specific Z slice")
    void testGetXPixelsInRegion() {
        // Pixels at Z=5: X coordinates are 10, 11, 12
        int[] result = microObjectWithParams.getXPixelsInRegion(5);

        assertThat(result).hasSize(3);
        assertThat(result).containsExactly(10, 11, 12);
    }

    @Test
    @DisplayName("getYPixelsInRegion() should return Y pixels in specific Z slice")
    void testGetYPixelsInRegion() {
        // Pixels at Z=5: Y coordinates are 20, 20, 20
        int[] result = microObjectWithParams.getYPixelsInRegion(5);

        assertThat(result).hasSize(3);
        assertThat(result).containsExactly(20, 20, 20);
    }

    @Test
    @DisplayName("getXPixelsInRegion() should return empty array for non-existent Z")
    void testGetXPixelsInRegionNonExistent() {
        int[] result = microObjectWithParams.getXPixelsInRegion(999);
        assertThat(result).isEmpty();
    }

    @Test
    @DisplayName("getYPixelsInRegion() should return empty array for non-existent Z")
    void testGetYPixelsInRegionNonExistent() {
        int[] result = microObjectWithParams.getYPixelsInRegion(999);
        assertThat(result).isEmpty();
    }

    // ========== Bounding Rectangle Tests ==========

    @Test
    @DisplayName("getBoundingRectangle() should calculate correct rectangle")
    void testGetBoundingRectangle() {
        Rectangle rect = microObjectWithParams.getBoundingRectangle();

        assertThat(rect).isNotNull();
        assertThat(rect.width).isEqualTo(2);   // X range: 12 - 10 = 2
        assertThat(rect.height).isEqualTo(2);  // Y range: 22 - 20 = 2
    }

    @Test
    @DisplayName("getBoundingRectangle() should center on centroid")
    void testGetBoundingRectangleCentered() {
        Rectangle rect = microObjectWithParams.getBoundingRectangle();

        // Centroid is at (11, 21), width=2, height=2
        // Start should be at centroid - (width/2, height/2) = (10, 20)
        assertThat(rect.x).isEqualTo(10);
        assertThat(rect.y).isEqualTo(20);
    }

    @Test
    @DisplayName("getBoundingRectangle() should handle single pixel")
    void testGetBoundingRectangleSinglePixel() {
        ArrayList<int[]> singlePixel = new ArrayList<>();
        singlePixel.add(new int[]{50});
        singlePixel.add(new int[]{100});
        singlePixel.add(new int[]{5});

        MicroObject single = new MicroObject(singlePixel, 0, testImageStacks, 1);
        Rectangle rect = single.getBoundingRectangle();

        assertThat(rect.width).isEqualTo(0);
        assertThat(rect.height).isEqualTo(0);
    }

    // ========== getResultPointer() Tests ==========

    @Test
    @DisplayName("getResultPointer() should return null initially")
    void testGetResultPointerInitially() {
        assertThat(microObject.getResultPointer()).isNull();
    }

    // ========== UnsupportedOperationException Tests ==========

    @Test
    @DisplayName("getObjectPixels() should throw UnsupportedOperationException")
    void testGetObjectPixelsThrowsException() {
        assertThatThrownBy(() -> microObject.getObjectPixels())
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    @Test
    @DisplayName("getRegions() should throw UnsupportedOperationException")
    void testGetRegionsThrowsException() {
        assertThatThrownBy(() -> microObject.getRegions())
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    @Test
    @DisplayName("getPixelCount() should throw UnsupportedOperationException")
    void testGetPixelCountThrowsException() {
        assertThatThrownBy(() -> microObject.getPixelCount())
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    @Test
    @DisplayName("getThresholdedIntegratedIntensity() should throw UnsupportedOperationException")
    void testGetThresholdedIntegratedIntensityThrowsException() {
        assertThatThrownBy(() -> microObject.getThresholdedIntegratedIntensity())
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    @Test
    @DisplayName("getThresholdedMeanIntensity() should throw UnsupportedOperationException")
    void testGetThresholdedMeanIntensityThrowsException() {
        assertThatThrownBy(() -> microObject.getThresholdedMeanIntensity())
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    @Test
    @DisplayName("setThreshold() should throw UnsupportedOperationException")
    void testSetThresholdThrowsException() {
        assertThatThrownBy(() -> microObject.setThreshold(0.5))
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    // ========== Inheritance Tests ==========

    @Test
    @DisplayName("Should implement Serializable")
    void testImplementsSerializable() {
        assertThat(microObject).isInstanceOf(java.io.Serializable.class);
    }

    @Test
    @DisplayName("Should implement MicroObjectModel")
    void testImplementsMicroObjectModel() {
        assertThat(microObject).isInstanceOf(MicroObjectModel.class);
    }

    // ========== Helper Methods ==========

    /**
     * Creates test pixels representing a small 3x3x3 cube
     * X: 10-12, Y: 20-22, Z: 5-7
     */
    private ArrayList<int[]> createTestPixels() {
        ArrayList<int[]> pixels = new ArrayList<>();

        int[] x = {10, 11, 12, 10, 11, 12, 10, 11, 12};
        int[] y = {20, 20, 20, 21, 21, 21, 22, 22, 22};
        int[] z = {5, 5, 5, 6, 6, 6, 7, 7, 7};

        pixels.add(x);
        pixels.add(y);
        pixels.add(z);

        return pixels;
    }

    /**
     * Creates test ImageStack array
     */
    private ImageStack[] createTestImageStacks(int width, int height, int slices) {
        ImageStack[] stacks = new ImageStack[1];
        ImageStack stack = new ImageStack(width, height);

        for (int i = 0; i < slices; i++) {
            ByteProcessor bp = new ByteProcessor(width, height);
            stack.addSlice(bp);
        }

        stacks[0] = stack;
        return stacks;
    }
}
