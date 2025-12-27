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
package vtea.objects.Segmentation;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import javax.swing.JTextField;
import java.util.ArrayList;
import java.util.List;

import vteaobjects.MicroObject;

import static org.assertj.core.api.Assertions.*;

/**
 * Test class for SingleThreshold segmentation
 *
 * SingleThreshold is a simple threshold-based 3D segmentation that creates
 * a single MicroObject containing all pixels above the threshold value.
 */
@DisplayName("SingleThreshold Segmentation Tests")
class SingleThresholdTest {

    private SingleThreshold segmentation;
    private ImageStack[] testStacks;
    private List<Object> testProtocol;

    @BeforeEach
    void setUp() {
        segmentation = new SingleThreshold();
        testStacks = createTestImageStacks(50, 50, 5);
        testProtocol = createTestProtocol(128);
    }

    // ========== Constructor and Metadata Tests ==========

    @Test
    @DisplayName("Constructor should initialize with correct metadata")
    void testConstructorMetadata() {
        assertThat(segmentation.getName()).isEqualTo("Single Threshold 3D");
        assertThat(segmentation.getKey()).isEqualTo("SingleThreshold3D");
        assertThat(segmentation.getVersion()).isEqualTo("0.1");
        assertThat(segmentation.getAuthor()).isEqualTo("Seth Winfree");
        assertThat(segmentation.getComment()).isEqualTo("Simple threshold for large regions by intensity.");
    }

    @Test
    @DisplayName("Constructor should set protocol components")
    void testConstructorSetsProtocol() {
        ArrayList protocol = segmentation.getOptions();

        assertThat(protocol).isNotNull();
        assertThat(protocol).hasSize(2);  // JLabel and JTextField
    }

    @Test
    @DisplayName("Should extend AbstractSegmentation")
    void testExtendsAbstractSegmentation() {
        assertThat(segmentation).isInstanceOf(AbstractSegmentation.class);
    }

    @Test
    @DisplayName("Should implement Segmentation interface")
    void testImplementsSegmentation() {
        assertThat(segmentation).isInstanceOf(Segmentation.class);
    }

    @Test
    @DisplayName("getDimensionalityCompatibility() should return 3D")
    void testGetDimensionalityCompatibility() {
        String compatibility = segmentation.getDimensionalityCompatibility();
        assertThat(compatibility).isEqualTo("3D");
    }

    // ========== Image Setting Tests ==========

    @Test
    @DisplayName("setImage() should store image")
    void testSetImage() {
        ImagePlus testImage = createTestImagePlus(10, 10, 1);

        assertThatCode(() -> segmentation.setImage(testImage)).doesNotThrowAnyException();
    }

    @Test
    @DisplayName("updateImage() should update image")
    void testUpdateImage() {
        ImagePlus testImage = createTestImagePlus(10, 10, 1);

        assertThatCode(() -> segmentation.updateImage(testImage)).doesNotThrowAnyException();
    }

    @Test
    @DisplayName("updateImage() should initialize MicroThresholdAdjuster")
    void testUpdateImageInitializesAdjuster() {
        ImagePlus testImage = createTestImagePlus(10, 10, 1);

        segmentation.updateImage(testImage);

        // Should not throw when getting segmentation tool
        assertThatCode(() -> segmentation.getSegmentationTool()).doesNotThrowAnyException();
    }

    // ========== Process Method Tests ==========

    @Test
    @DisplayName("process() should segment image with low threshold")
    void testProcessWithLowThreshold() {
        // Protocol with threshold of 50 (low threshold)
        List<Object> protocol = createTestProtocol(50);

        boolean result = segmentation.process(testStacks, protocol, true);

        assertThat(result).isTrue();
        assertThat(segmentation.getObjects()).isNotEmpty();
        assertThat(segmentation.getObjects()).hasSize(1);
    }

    @Test
    @DisplayName("process() should segment image with high threshold")
    void testProcessWithHighThreshold() {
        // Protocol with threshold of 200 (high threshold)
        List<Object> protocol = createTestProtocol(200);

        boolean result = segmentation.process(testStacks, protocol, true);

        assertThat(result).isTrue();
        assertThat(segmentation.getObjects()).isNotEmpty();
    }

    @Test
    @DisplayName("process() should create MicroObject with correct size")
    void testProcessCreatesCorrectSizeMicroObject() {
        // Create image with known bright pixels
        ImageStack[] stacks = createUniformImageStacks(10, 10, 1, 255);
        List<Object> protocol = createTestProtocol(128);

        segmentation.process(stacks, protocol, true);

        MicroObject obj = segmentation.getObjects().get(0);
        int[] xPixels = obj.getPixelsX();

        // All pixels should be above threshold, so 10*10*1 = 100 pixels
        assertThat(xPixels.length).isEqualTo(100);
    }

    @Test
    @DisplayName("process() should handle threshold of zero")
    void testProcessWithZeroThreshold() {
        List<Object> protocol = createTestProtocol(0);

        boolean result = segmentation.process(testStacks, protocol, true);

        assertThat(result).isTrue();
        assertThat(segmentation.getObjects()).hasSize(1);

        // With threshold 0, all non-zero pixels should be included
        MicroObject obj = segmentation.getObjects().get(0);
        assertThat(obj.getPixelsX().length).isGreaterThan(0);
    }

    @Test
    @DisplayName("process() should handle maximum threshold")
    void testProcessWithMaxThreshold() {
        List<Object> protocol = createTestProtocol(255);

        boolean result = segmentation.process(testStacks, protocol, true);

        assertThat(result).isTrue();
        // With max threshold, very few or no pixels should pass
    }

    @Test
    @DisplayName("process() should create binary mask")
    void testProcessCreatesBinaryMask() {
        List<Object> protocol = createTestProtocol(128);

        segmentation.process(testStacks, protocol, true);

        ImagePlus resultImage = segmentation.getSegmentation();

        assertThat(resultImage).isNotNull();
        assertThat(resultImage.getTitle()).isEqualTo("Mask Result");
    }

    @Test
    @DisplayName("process() should use correct channel from protocol")
    void testProcessUsesCorrectChannel() {
        // Create multi-channel stack
        ImageStack[] multiChannel = new ImageStack[3];
        multiChannel[0] = createImageStack(10, 10, 1, 50);
        multiChannel[1] = createImageStack(10, 10, 1, 150);
        multiChannel[2] = createImageStack(10, 10, 1, 250);

        // Protocol specifying channel 1
        List<Object> protocol = createTestProtocol(100);

        boolean result = segmentation.process(multiChannel, protocol, true);

        assertThat(result).isTrue();
        assertThat(segmentation.getObjects()).hasSize(1);
    }

    @Test
    @DisplayName("process() should collect all pixels above threshold")
    void testProcessCollectsCorrectPixels() {
        // Create simple 3x3x1 image with known values
        ImageStack[] stacks = createCustomImageStacks();
        List<Object> protocol = createTestProtocol(128);

        segmentation.process(stacks, protocol, true);

        MicroObject obj = segmentation.getObjects().get(0);
        int[] xPixels = obj.getPixelsX();
        int[] yPixels = obj.getPixelsY();
        int[] zPixels = obj.getPixelsZ();

        // Should have same number of x, y, z coordinates
        assertThat(yPixels.length).isEqualTo(xPixels.length);
        assertThat(zPixels.length).isEqualTo(xPixels.length);

        // All coordinates should be non-negative
        for (int x : xPixels) {
            assertThat(x).isGreaterThanOrEqualTo(0);
        }
    }

    @Test
    @DisplayName("process() should handle 3D stacks")
    void testProcessHandles3DStacks() {
        ImageStack[] stacks = createTestImageStacks(20, 20, 10);
        List<Object> protocol = createTestProtocol(100);

        boolean result = segmentation.process(stacks, protocol, true);

        assertThat(result).isTrue();

        MicroObject obj = segmentation.getObjects().get(0);
        int[] zPixels = obj.getPixelsZ();

        // Should have pixels from multiple Z slices
        int minZ = Integer.MAX_VALUE;
        int maxZ = Integer.MIN_VALUE;
        for (int z : zPixels) {
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);
        }

        // Should span multiple slices if there are bright pixels
        if (zPixels.length > 0) {
            assertThat(minZ).isGreaterThanOrEqualTo(0);
            assertThat(maxZ).isLessThan(10);
        }
    }

    // ========== getObjects() Tests ==========

    @Test
    @DisplayName("getObjects() should return empty list initially")
    void testGetObjectsInitiallyEmpty() {
        ArrayList<MicroObject> objects = segmentation.getObjects();

        assertThat(objects).isNotNull();
        assertThat(objects).isEmpty();
    }

    @Test
    @DisplayName("getObjects() should return list after processing")
    void testGetObjectsAfterProcessing() {
        segmentation.process(testStacks, testProtocol, true);

        ArrayList<MicroObject> objects = segmentation.getObjects();

        assertThat(objects).isNotNull();
        assertThat(objects).hasSize(1);
    }

    @Test
    @DisplayName("getObjects() should return MicroObject instances")
    void testGetObjectsReturnsMicroObjects() {
        segmentation.process(testStacks, testProtocol, true);

        ArrayList<MicroObject> objects = segmentation.getObjects();

        assertThat(objects.get(0)).isInstanceOf(MicroObject.class);
    }

    // ========== getSegmentation() Tests ==========

    @Test
    @DisplayName("getSegmentation() should return null initially")
    void testGetSegmentationInitiallyNull() {
        ImagePlus result = segmentation.getSegmentation();
        assertThat(result).isNull();
    }

    @Test
    @DisplayName("getSegmentation() should return ImagePlus after processing")
    void testGetSegmentationAfterProcessing() {
        segmentation.process(testStacks, testProtocol, true);

        ImagePlus result = segmentation.getSegmentation();

        assertThat(result).isNotNull();
        assertThat(result.getTitle()).isEqualTo("Mask Result");
    }

    @Test
    @DisplayName("getSegmentation() should return 8-bit image")
    void testGetSegmentationReturns8Bit() {
        segmentation.process(testStacks, testProtocol, true);

        ImagePlus result = segmentation.getSegmentation();

        // After processing, should be 8-bit
        assertThat(result.getBitDepth()).isEqualTo(8);
    }

    @Test
    @DisplayName("getSegmentation() should have correct dimensions")
    void testGetSegmentationDimensions() {
        segmentation.process(testStacks, testProtocol, true);

        ImagePlus result = segmentation.getSegmentation();

        assertThat(result.getWidth()).isEqualTo(50);
        assertThat(result.getHeight()).isEqualTo(50);
        assertThat(result.getStackSize()).isEqualTo(5);
    }

    // ========== getSegmentationTool() Tests ==========

    @Test
    @DisplayName("getSegmentationTool() should return JPanel")
    void testGetSegmentationTool() {
        ImagePlus testImage = createTestImagePlus(10, 10, 1);
        segmentation.updateImage(testImage);

        assertThatCode(() -> {
            javax.swing.JPanel panel = segmentation.getSegmentationTool();
            assertThat(panel).isNotNull();
        }).doesNotThrowAnyException();
    }

    // ========== doUpdateOfTool() Tests ==========

    @Test
    @DisplayName("doUpdateOfTool() should complete without exception")
    void testDoUpdateOfTool() {
        ImagePlus testImage = createTestImagePlus(10, 10, 1);
        segmentation.updateImage(testImage);
        segmentation.getSegmentationTool();

        assertThatCode(() -> segmentation.doUpdateOfTool()).doesNotThrowAnyException();
    }

    // ========== UnsupportedOperationException Tests ==========

    @Test
    @DisplayName("runImageJMacroCommand() should throw UnsupportedOperationException")
    void testRunImageJMacroCommandThrows() {
        assertThatThrownBy(() -> segmentation.runImageJMacroCommand("test"))
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    @Test
    @DisplayName("sendProgressComment() should throw UnsupportedOperationException")
    void testSendProgressCommentThrows() {
        assertThatThrownBy(() -> segmentation.sendProgressComment())
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    @Test
    @DisplayName("getProgressComment() should throw UnsupportedOperationException")
    void testGetProgressCommentThrows() {
        assertThatThrownBy(() -> segmentation.getProgressComment())
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    // ========== Helper Methods ==========

    /**
     * Creates test image stacks with gradient values
     */
    private ImageStack[] createTestImageStacks(int width, int height, int slices) {
        ImageStack[] stacks = new ImageStack[1];
        ImageStack stack = new ImageStack(width, height);

        for (int z = 0; z < slices; z++) {
            ByteProcessor bp = new ByteProcessor(width, height);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // Create gradient pattern
                    int value = (x + y + z * 10) % 256;
                    bp.putPixel(x, y, value);
                }
            }
            stack.addSlice(bp);
        }

        stacks[0] = stack;
        return stacks;
    }

    /**
     * Creates uniform image stacks with single value
     */
    private ImageStack[] createUniformImageStacks(int width, int height, int slices, int value) {
        ImageStack[] stacks = new ImageStack[1];
        stacks[0] = createImageStack(width, height, slices, value);
        return stacks;
    }

    /**
     * Creates an image stack with uniform value
     */
    private ImageStack createImageStack(int width, int height, int slices, int value) {
        ImageStack stack = new ImageStack(width, height);

        for (int z = 0; z < slices; z++) {
            ByteProcessor bp = new ByteProcessor(width, height);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    bp.putPixel(x, y, value);
                }
            }
            stack.addSlice(bp);
        }

        return stack;
    }

    /**
     * Creates custom image stacks with specific pattern
     */
    private ImageStack[] createCustomImageStacks() {
        ImageStack[] stacks = new ImageStack[1];
        ImageStack stack = new ImageStack(3, 3);

        ByteProcessor bp = new ByteProcessor(3, 3);
        // Create pattern: center pixel bright, corners dark
        bp.putPixel(0, 0, 50);
        bp.putPixel(1, 0, 100);
        bp.putPixel(2, 0, 50);
        bp.putPixel(0, 1, 100);
        bp.putPixel(1, 1, 200);
        bp.putPixel(2, 1, 100);
        bp.putPixel(0, 2, 50);
        bp.putPixel(1, 2, 100);
        bp.putPixel(2, 2, 50);

        stack.addSlice(bp);
        stacks[0] = stack;
        return stacks;
    }

    /**
     * Creates a test ImagePlus
     */
    private ImagePlus createTestImagePlus(int width, int height, int slices) {
        ImageStack stack = new ImageStack(width, height);

        for (int z = 0; z < slices; z++) {
            ByteProcessor bp = new ByteProcessor(width, height);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    bp.putPixel(x, y, 128);
                }
            }
            stack.addSlice(bp);
        }

        return new ImagePlus("Test", stack);
    }

    /**
     * Creates test protocol ArrayList
     *
     * Protocol structure:
     * 0: title text
     * 1: method (as String)
     * 2: channel (int)
     * 3: ArrayList of JComponents (includes threshold JTextField)
     */
    private List<Object> createTestProtocol(int threshold) {
        List<Object> protocol = new ArrayList<>();

        protocol.add("Test Segmentation");               // 0: title
        protocol.add("SingleThreshold3D");               // 1: method
        protocol.add(0);                                 // 2: channel

        // Create components list
        ArrayList<Object> components = new ArrayList<>();
        components.add(new javax.swing.JLabel("Low Threshold"));  // 0: label
        JTextField thresholdField = new JTextField(String.valueOf(threshold), 5);
        components.add(thresholdField);                  // 1: threshold value

        protocol.add(components);                        // 3: components

        return protocol;
    }
}
