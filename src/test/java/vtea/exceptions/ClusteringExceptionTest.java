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
 * Tests for ClusteringException.
 *
 * @author VTEA Development Team
 */
@DisplayName("ClusteringException Tests")
class ClusteringExceptionTest extends BaseTest {

    @Test
    @DisplayName("Should extend VTEAException")
    void testInheritance() {
        // When
        ClusteringException exception = new ClusteringException();

        // Then
        assertThat(exception).isInstanceOf(VTEAException.class);
        assertThat(exception).isInstanceOf(Exception.class);
    }

    @Test
    @DisplayName("Should create with default constructor")
    void testDefaultConstructor() {
        // When
        ClusteringException exception = new ClusteringException();

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isNull();
    }

    @Test
    @DisplayName("Should create with message")
    void testMessageConstructor() {
        // Given
        String message = "K-means clustering failed to converge";

        // When
        ClusteringException exception = new ClusteringException(message);

        // Then
        assertThat(exception.getMessage()).isEqualTo(message);
    }

    @Test
    @DisplayName("Should create with message and cause")
    void testMessageAndCauseConstructor() {
        // Given
        String message = "Hierarchical clustering failed";
        Throwable cause = new IllegalArgumentException("Invalid distance metric");

        // When
        ClusteringException exception = new ClusteringException(message, cause);

        // Then
        assertThat(exception.getMessage()).isEqualTo(message);
        assertThat(exception.getCause()).isEqualTo(cause);
    }

    @Test
    @DisplayName("Should create with cause only")
    void testCauseConstructor() {
        // Given
        Throwable cause = new ArithmeticException("Division by zero in covariance matrix");

        // When
        ClusteringException exception = new ClusteringException(cause);

        // Then
        assertThat(exception.getCause()).isEqualTo(cause);
    }

    @Test
    @DisplayName("Should be throwable for clustering errors")
    void testThrowable() {
        // Then
        assertThatThrownBy(() -> {
            throw new ClusteringException("Gaussian mixture model failed");
        })
        .isInstanceOf(ClusteringException.class)
        .isInstanceOf(VTEAException.class)
        .hasMessageContaining("Gaussian");
    }

    @Test
    @DisplayName("Should support real-world clustering error scenario")
    void testRealWorldScenario() {
        // Given
        String algorithm = "K-means";
        String issue = "Empty cluster detected";
        int iteration = 42;

        // When
        ClusteringException exception = new ClusteringException(
            String.format("%s failed at iteration %d: %s", algorithm, iteration, issue)
        );

        // Then
        assertThat(exception.getMessage())
            .contains(algorithm)
            .contains("42")
            .contains("Empty cluster");
    }

    @Test
    @DisplayName("Should wrap dimensionality reduction errors")
    void testDimensionalityReductionError() {
        // Given
        String message = "t-SNE dimensionality reduction failed";
        Throwable cause = new OutOfMemoryError("Insufficient memory for distance matrix");

        // When
        ClusteringException exception = new ClusteringException(message, cause);

        // Then
        assertThat(exception.getMessage()).contains("t-SNE");
        assertThat(exception.getCause()).isInstanceOf(OutOfMemoryError.class);
    }

    @Test
    @DisplayName("Should be catchable as VTEAException")
    void testCatchAsVTEAException() {
        // Then
        assertThatThrownBy(() -> {
            throw new ClusteringException("Test");
        })
        .isInstanceOf(VTEAException.class);
    }

    @Test
    @DisplayName("Should support model selection errors")
    void testModelSelectionError() {
        // Given
        String error = "BIC calculation failed - singular covariance matrix";

        // When
        ClusteringException exception = new ClusteringException(error);

        // Then
        assertThat(exception.getMessage())
            .contains("BIC")
            .contains("singular");
    }

    @Test
    @DisplayName("Should handle invalid cluster count")
    void testInvalidClusterCount() {
        // Given
        int requestedK = 0;
        int dataPoints = 100;

        // When
        ClusteringException exception = new ClusteringException(
            String.format("Invalid cluster count: K=%d for %d data points", requestedK, dataPoints)
        );

        // Then
        assertThat(exception.getMessage())
            .contains("K=0")
            .contains("100 data points");
    }

    @Test
    @DisplayName("Should wrap SMILE library exceptions")
    void testSMILEExceptionWrapping() {
        // Given - Simulating a SMILE library exception
        RuntimeException smileException = new RuntimeException("SMILE: Matrix is singular");

        // When
        ClusteringException exception = new ClusteringException(
            "Clustering algorithm failed", smileException
        );

        // Then
        assertThat(exception.getCause().getMessage()).contains("SMILE");
        assertThat(exception.getCause().getMessage()).contains("singular");
    }
}
