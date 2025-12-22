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
package vtea.processor;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.*;

/**
 * Test class for SegmentationProcessor
 *
 * Tests the SegmentationProcessor which handles asynchronous segmentation
 * of images using the SwingWorker pattern and ForkJoinPool for parallel processing.
 */
@DisplayName("SegmentationProcessor Tests")
@Timeout(10)
class SegmentationProcessorTest {

    private SegmentationProcessor processor;
    private SegmentationProcessor processorWithParams;
    private ImagePlus testImage;
    private ArrayList<Object> testProtocol;
    private MockPropertyChangeListener propertyListener;

    @BeforeEach
    void setUp() {
        // Create default processor
        processor = new SegmentationProcessor();

        // Create test image
        testImage = createTestImage(100, 100, 1);

        // Create test protocol
        testProtocol = createTestProtocol();

        // Create processor with parameters
        processorWithParams = new SegmentationProcessor("testKey", testImage, testProtocol);

        // Create property change listener
        propertyListener = new MockPropertyChangeListener();
    }

    // ========== Metadata Tests ==========

    @Test
    @DisplayName("Default constructor should initialize with correct metadata")
    void testDefaultConstructorMetadata() {
        assertThat(processor.getName()).isEqualTo("Segmentation Processor");
        assertThat(processor.getKey()).isEqualTo("SegmentationProcessor");
        assertThat(processor.VERSION).isEqualTo("0.0");
        assertThat(processor.AUTHOR).isEqualTo("Seth Winfree");
        assertThat(processor.COMMENT).isEqualTo("Processor for segmentation processing");
    }

    @Test
    @DisplayName("Parameterized constructor should initialize with correct metadata")
    void testParameterizedConstructorMetadata() {
        assertThat(processorWithParams.getName()).isEqualTo("Segmentation Processor");
        assertThat(processorWithParams.getKey()).isEqualTo("SegmentationProcessor");
        assertThat(processorWithParams.VERSION).isEqualTo("0.0");
        assertThat(processorWithParams.AUTHOR).isEqualTo("Seth Winfree");
        assertThat(processorWithParams.COMMENT).isEqualTo("Processor for segmentation processing");
    }

    @Test
    @DisplayName("Parameterized constructor should store UID key")
    void testParameterizedConstructorStoresUIDKey() {
        assertThat(processorWithParams.getUIDKey()).isEqualTo("testKey");
    }

    // ========== Constructor Tests ==========

    @Test
    @DisplayName("Default constructor should create valid instance")
    void testDefaultConstructor() {
        assertThat(processor).isNotNull();
        assertThat(processor.getProtocol()).isNull();
        assertThat(processor.getObjects()).isNotNull();
        assertThat(processor.getObjects()).isEmpty();
    }

    @Test
    @DisplayName("Parameterized constructor should store protocol")
    void testParameterizedConstructorStoresProtocol() {
        ArrayList protocol = processorWithParams.getProtocol();

        assertThat(protocol).isNotNull();
        assertThat(protocol).isSameAs(testProtocol);
    }

    @Test
    @DisplayName("Should handle null protocol in parameterized constructor")
    void testParameterizedConstructorWithNullProtocol() {
        SegmentationProcessor proc = new SegmentationProcessor("key", testImage, null);

        assertThat(proc.getProtocol()).isNull();
        assertThat(proc.getUIDKey()).isEqualTo("key");
    }

    @Test
    @DisplayName("Should handle null image in parameterized constructor")
    void testParameterizedConstructorWithNullImage() {
        SegmentationProcessor proc = new SegmentationProcessor("key", null, testProtocol);

        assertThat(proc.getProtocol()).isSameAs(testProtocol);
        assertThat(proc.getUIDKey()).isEqualTo("key");
    }

    // ========== getObjects() Tests ==========

    @Test
    @DisplayName("getObjects() should return empty list initially")
    void testGetObjectsInitiallyEmpty() {
        ArrayList objects = processor.getObjects();

        assertThat(objects).isNotNull();
        assertThat(objects).isEmpty();
    }

    @Test
    @DisplayName("getObjects() should return ArrayList type")
    void testGetObjectsReturnsArrayList() {
        ArrayList objects = processor.getObjects();

        assertThat(objects).isInstanceOf(ArrayList.class);
    }

    @Test
    @DisplayName("getObjects() should return synchronized list")
    void testGetObjectsReturnsSynchronizedList() {
        // The volumes list is created as Collections.synchronizedList
        ArrayList objects = processor.getObjects();

        assertThat(objects).isNotNull();
        // Synchronized lists are thread-safe
        assertThat(objects.getClass().getName()).contains("Synchronized");
    }

    // ========== getProtocol() Tests ==========

    @Test
    @DisplayName("getProtocol() should return null for default constructor")
    void testGetProtocolReturnsNullForDefaultConstructor() {
        assertThat(processor.getProtocol()).isNull();
    }

    @Test
    @DisplayName("getProtocol() should return stored protocol")
    void testGetProtocolReturnsStoredProtocol() {
        ArrayList protocol = processorWithParams.getProtocol();

        assertThat(protocol).isNotNull();
        assertThat(protocol).hasSize(5);
        assertThat(protocol.get(0)).isEqualTo("Test Title");
    }

    @Test
    @DisplayName("getProtocol() should return same instance")
    void testGetProtocolReturnsSameInstance() {
        ArrayList protocol1 = processorWithParams.getProtocol();
        ArrayList protocol2 = processorWithParams.getProtocol();

        assertThat(protocol1).isSameAs(protocol2);
    }

    // ========== FireProgressChange() Tests ==========

    @Test
    @DisplayName("FireProgressChange() should fire progress property change")
    void testFireProgressChangeFiresProgressEvent() throws Exception {
        processorWithParams.addPropertyChangeListener(propertyListener);

        processorWithParams.FireProgressChange("test message", 50.0);

        // Give time for async property change
        Thread.sleep(50);

        assertThat(propertyListener.hasProgressEvent()).isTrue();
        assertThat(propertyListener.getLastProgressValue()).isEqualTo(50);
    }

    @Test
    @DisplayName("FireProgressChange() should fire comment property change")
    void testFireProgressChangeFiresCommentEvent() throws Exception {
        processorWithParams.addPropertyChangeListener(propertyListener);

        processorWithParams.FireProgressChange("test message", 50.0);

        // Give time for async property change
        Thread.sleep(50);

        assertThat(propertyListener.hasCommentEvent()).isTrue();
        assertThat(propertyListener.getLastCommentMessage()).isEqualTo("test message");
    }

    @Test
    @DisplayName("FireProgressChange() should handle zero progress")
    void testFireProgressChangeWithZeroProgress() throws Exception {
        processorWithParams.addPropertyChangeListener(propertyListener);

        processorWithParams.FireProgressChange("starting", 0.0);

        Thread.sleep(50);

        assertThat(propertyListener.getLastProgressValue()).isEqualTo(0);
    }

    @Test
    @DisplayName("FireProgressChange() should handle 100% progress")
    void testFireProgressChangeWith100Progress() throws Exception {
        processorWithParams.addPropertyChangeListener(propertyListener);

        processorWithParams.FireProgressChange("complete", 100.0);

        Thread.sleep(50);

        assertThat(propertyListener.getLastProgressValue()).isEqualTo(100);
    }

    @Test
    @DisplayName("FireProgressChange() should convert double to int for progress")
    void testFireProgressChangeConvertsDoubleToInt() throws Exception {
        processorWithParams.addPropertyChangeListener(propertyListener);

        processorWithParams.FireProgressChange("test", 75.7);

        Thread.sleep(50);

        // Should be converted to int (75)
        assertThat(propertyListener.getLastProgressValue()).isEqualTo(75);
    }

    // ========== UnsupportedOperationException Tests ==========

    @Test
    @DisplayName("process(ArrayList, String...) should throw UnsupportedOperationException")
    void testProcessThrowsUnsupportedException() {
        assertThatThrownBy(() -> processor.process(new ArrayList(), "test"))
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    @Test
    @DisplayName("getChange() should throw UnsupportedOperationException")
    void testGetChangeThrowsUnsupportedException() {
        assertThatThrownBy(() -> processor.getChange())
            .isInstanceOf(UnsupportedOperationException.class)
            .hasMessageContaining("Not supported yet");
    }

    // ========== Inheritance Tests ==========

    @Test
    @DisplayName("Should extend AbstractProcessor")
    void testExtendsAbstractProcessor() {
        assertThat(processor).isInstanceOf(AbstractProcessor.class);
    }

    @Test
    @DisplayName("Should implement Processor interface")
    void testImplementsProcessor() {
        assertThat(processor).isInstanceOf(Processor.class);
    }

    @Test
    @DisplayName("Should be a SwingWorker")
    void testIsSwingWorker() {
        assertThat(processor).isInstanceOf(javax.swing.SwingWorker.class);
    }

    // ========== done() Method Tests ==========

    @Test
    @DisplayName("done() method should complete without exception")
    void testDoneMethodCompletes() {
        assertThatCode(() -> processor.done()).doesNotThrowAnyException();
    }

    // ========== Helper Methods ==========

    /**
     * Creates a simple test image
     */
    private ImagePlus createTestImage(int width, int height, int slices) {
        ImageStack stack = new ImageStack(width, height);

        for (int i = 0; i < slices; i++) {
            ByteProcessor bp = new ByteProcessor(width, height);
            // Fill with some test data
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    bp.putPixel(x, y, (x + y) % 256);
                }
            }
            stack.addSlice(bp);
        }

        return new ImagePlus("Test Image", stack);
    }

    /**
     * Creates a test protocol ArrayList
     *
     * Protocol structure:
     * 0: title text
     * 1: method (as String)
     * 2: channel
     * 3: ArrayList of JComponents
     * 4: ArrayList of ArrayList for morphology
     */
    private ArrayList<Object> createTestProtocol() {
        ArrayList<Object> protocol = new ArrayList<>();
        protocol.add("Test Title");                    // 0: title
        protocol.add("TestMethod");                    // 1: method
        protocol.add(0);                               // 2: channel
        protocol.add(new ArrayList<>());               // 3: components
        protocol.add(new ArrayList<ArrayList>());      // 4: morphologies

        return protocol;
    }

    /**
     * Mock PropertyChangeListener to capture events
     */
    private static class MockPropertyChangeListener implements PropertyChangeListener {
        private boolean hasProgressEvent = false;
        private boolean hasCommentEvent = false;
        private int lastProgressValue = -1;
        private String lastCommentMessage = null;
        private String lastCommentKey = null;

        @Override
        public void propertyChange(PropertyChangeEvent evt) {
            if ("progress".equals(evt.getPropertyName())) {
                hasProgressEvent = true;
                lastProgressValue = (Integer) evt.getNewValue();
            } else if ("comment".equals(evt.getPropertyName())) {
                hasCommentEvent = true;
                lastCommentKey = (String) evt.getOldValue();
                lastCommentMessage = (String) evt.getNewValue();
            }
        }

        public boolean hasProgressEvent() {
            return hasProgressEvent;
        }

        public boolean hasCommentEvent() {
            return hasCommentEvent;
        }

        public int getLastProgressValue() {
            return lastProgressValue;
        }

        public String getLastCommentMessage() {
            return lastCommentMessage;
        }

        public String getLastCommentKey() {
            return lastCommentKey;
        }
    }
}
