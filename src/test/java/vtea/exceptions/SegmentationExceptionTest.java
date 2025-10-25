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
package vtea.exceptions;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import vtea.BaseTest;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for SegmentationException.
 *
 * @author VTEA Development Team
 */
@DisplayName("SegmentationException Tests")
class SegmentationExceptionTest extends BaseTest {

    @Test
    @DisplayName("Should extend VTEAException")
    void testInheritance() {
        // When
        SegmentationException exception = new SegmentationException();

        // Then
        assertThat(exception).isInstanceOf(VTEAException.class);
        assertThat(exception).isInstanceOf(Exception.class);
    }

    @Test
    @DisplayName("Should create with default constructor")
    void testDefaultConstructor() {
        // When
        SegmentationException exception = new SegmentationException();

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isNull();
    }

    @Test
    @DisplayName("Should create with message")
    void testMessageConstructor() {
        // Given
        String message = "Object detection failed";

        // When
        SegmentationException exception = new SegmentationException(message);

        // Then
        assertThat(exception.getMessage()).isEqualTo(message);
    }

    @Test
    @DisplayName("Should create with message and cause")
    void testMessageAndCauseConstructor() {
        // Given
        String message = "3D segmentation failed";
        Throwable cause = new OutOfMemoryError("Insufficient memory for volume");

        // When
        SegmentationException exception = new SegmentationException(message, cause);

        // Then
        assertThat(exception.getMessage()).isEqualTo(message);
        assertThat(exception.getCause()).isEqualTo(cause);
    }

    @Test
    @DisplayName("Should create with cause only")
    void testCauseConstructor() {
        // Given
        Throwable cause = new IllegalStateException("Invalid threshold");

        // When
        SegmentationException exception = new SegmentationException(cause);

        // Then
        assertThat(exception.getCause()).isEqualTo(cause);
    }

    @Test
    @DisplayName("Should be throwable for segmentation errors")
    void testThrowable() {
        // Then
        assertThatThrownBy(() -> {
            throw new SegmentationException("No objects detected");
        })
        .isInstanceOf(SegmentationException.class)
        .isInstanceOf(VTEAException.class)
        .hasMessageContaining("No objects");
    }

    @Test
    @DisplayName("Should support real-world segmentation error scenario")
    void testRealWorldScenario() {
        // Given
        String algorithm = "LayerCake3D";
        String issue = "No nuclei found in volume";

        // When
        SegmentationException exception = new SegmentationException(
            algorithm + " segmentation failed: " + issue
        );

        // Then
        assertThat(exception.getMessage())
            .contains(algorithm)
            .contains(issue);
    }

    @Test
    @DisplayName("Should wrap algorithm-specific exceptions")
    void testExceptionWrapping() {
        // Given
        NullPointerException lowLevel = new NullPointerException("Image stack is null");

        // When
        SegmentationException exception = new SegmentationException(
            "Segmentation preprocessing failed", lowLevel
        );

        // Then
        assertThat(exception.getCause()).isEqualTo(lowLevel);
        assertThat(exception.getCause().getMessage()).contains("null");
    }

    @Test
    @DisplayName("Should be catchable as VTEAException")
    void testCatchAsVTEAException() {
        // Then
        assertThatThrownBy(() -> {
            throw new SegmentationException("Test");
        })
        .isInstanceOf(VTEAException.class);
    }

    @Test
    @DisplayName("Should support error context for debugging")
    void testErrorContext() {
        // Given
        String context = "Failed at slice 42 of 100";

        // When
        SegmentationException exception = new SegmentationException(
            "FloodFill3D failed: " + context
        );

        // Then
        assertThat(exception.getMessage()).contains("slice 42");
    }
}
