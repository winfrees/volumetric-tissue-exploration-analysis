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
 * Tests for ImageProcessingException.
 *
 * @author VTEA Development Team
 */
@DisplayName("ImageProcessingException Tests")
class ImageProcessingExceptionTest extends BaseTest {

    @Test
    @DisplayName("Should extend VTEAException")
    void testInheritance() {
        // When
        ImageProcessingException exception = new ImageProcessingException();

        // Then
        assertThat(exception).isInstanceOf(VTEAException.class);
        assertThat(exception).isInstanceOf(Exception.class);
    }

    @Test
    @DisplayName("Should create with default constructor")
    void testDefaultConstructor() {
        // When
        ImageProcessingException exception = new ImageProcessingException();

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isNull();
    }

    @Test
    @DisplayName("Should create with message")
    void testMessageConstructor() {
        // Given
        String message = "Image filtering failed";

        // When
        ImageProcessingException exception = new ImageProcessingException(message);

        // Then
        assertThat(exception.getMessage()).isEqualTo(message);
    }

    @Test
    @DisplayName("Should create with message and cause")
    void testMessageAndCauseConstructor() {
        // Given
        String message = "Gaussian filter failed";
        Throwable cause = new IllegalArgumentException("Invalid sigma");

        // When
        ImageProcessingException exception = new ImageProcessingException(message, cause);

        // Then
        assertThat(exception.getMessage()).isEqualTo(message);
        assertThat(exception.getCause()).isEqualTo(cause);
    }

    @Test
    @DisplayName("Should create with cause only")
    void testCauseConstructor() {
        // Given
        Throwable cause = new OutOfMemoryError("Image too large");

        // When
        ImageProcessingException exception = new ImageProcessingException(cause);

        // Then
        assertThat(exception.getCause()).isEqualTo(cause);
    }

    @Test
    @DisplayName("Should be throwable for image processing errors")
    void testThrowable() {
        // Then
        assertThatThrownBy(() -> {
            throw new ImageProcessingException("Threshold calculation failed");
        })
        .isInstanceOf(ImageProcessingException.class)
        .isInstanceOf(VTEAException.class)
        .hasMessageContaining("Threshold");
    }

    @Test
    @DisplayName("Should support real-world image processing error scenario")
    void testRealWorldScenario() {
        // Given
        String operation = "Median filter";
        String details = "Kernel size too large for image dimensions";

        // When
        ImageProcessingException exception = new ImageProcessingException(
            operation + " failed: " + details
        );

        // Then
        assertThat(exception.getMessage())
            .contains(operation)
            .contains(details);
    }

    @Test
    @DisplayName("Should wrap low-level exceptions")
    void testExceptionWrapping() {
        // Given
        ArrayIndexOutOfBoundsException lowLevel =
            new ArrayIndexOutOfBoundsException("Pixel access out of bounds");

        // When
        ImageProcessingException exception = new ImageProcessingException(
            "Image processing failed", lowLevel
        );

        // Then
        assertThat(exception.getCause()).isEqualTo(lowLevel);
        assertThat(exception.getCause().getMessage()).contains("out of bounds");
    }

    @Test
    @DisplayName("Should be catchable as VTEAException")
    void testCatchAsVTEAException() {
        // Then
        assertThatThrownBy(() -> {
            throw new ImageProcessingException("Test");
        })
        .isInstanceOf(VTEAException.class);
    }

    @Test
    @DisplayName("Should support null values")
    void testNullValues() {
        // When
        ImageProcessingException exception1 = new ImageProcessingException((String) null);
        ImageProcessingException exception2 = new ImageProcessingException("msg", null);

        // Then
        assertThat(exception1.getMessage()).isNull();
        assertThat(exception2.getCause()).isNull();
    }
}
