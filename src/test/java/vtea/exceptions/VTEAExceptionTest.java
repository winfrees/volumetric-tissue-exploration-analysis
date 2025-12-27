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

import java.io.*;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for VTEAException base class.
 *
 * @author VTEA Development Team
 */
@DisplayName("VTEAException Tests")
class VTEAExceptionTest extends BaseTest {

    @Test
    @DisplayName("Should create exception with default constructor")
    void testDefaultConstructor() {
        // When
        VTEAException exception = new VTEAException();

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isNull();
        assertThat(exception.getCause()).isNull();
    }

    @Test
    @DisplayName("Should create exception with message")
    void testMessageConstructor() {
        // Given
        String message = "Test error message";

        // When
        VTEAException exception = new VTEAException(message);

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isEqualTo(message);
        assertThat(exception.getCause()).isNull();
    }

    @Test
    @DisplayName("Should create exception with message and cause")
    void testMessageAndCauseConstructor() {
        // Given
        String message = "Test error message";
        Throwable cause = new RuntimeException("Root cause");

        // When
        VTEAException exception = new VTEAException(message, cause);

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isEqualTo(message);
        assertThat(exception.getCause()).isEqualTo(cause);
    }

    @Test
    @DisplayName("Should create exception with cause only")
    void testCauseConstructor() {
        // Given
        Throwable cause = new RuntimeException("Root cause");

        // When
        VTEAException exception = new VTEAException(cause);

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getCause()).isEqualTo(cause);
        assertThat(exception.getMessage()).contains("RuntimeException");
    }

    @Test
    @DisplayName("Should preserve exception message")
    void testMessagePreservation() {
        // Given
        String message = "Detailed error description";

        // When
        VTEAException exception = new VTEAException(message);

        // Then
        assertThat(exception.toString()).contains(message);
        assertThat(exception.getLocalizedMessage()).isEqualTo(message);
    }

    @Test
    @DisplayName("Should preserve cause chain")
    void testCauseChainPreservation() {
        // Given
        Throwable rootCause = new IOException("File not found");
        Throwable intermediateCause = new RuntimeException("Processing failed", rootCause);
        String message = "VTEA operation failed";

        // When
        VTEAException exception = new VTEAException(message, intermediateCause);

        // Then
        assertThat(exception.getCause()).isEqualTo(intermediateCause);
        assertThat(exception.getCause().getCause()).isEqualTo(rootCause);
    }

    @Test
    @DisplayName("Should be throwable")
    void testThrowable() {
        // Given
        String message = "Test exception";

        // Then
        assertThatThrownBy(() -> {
            throw new VTEAException(message);
        })
        .isInstanceOf(VTEAException.class)
        .isInstanceOf(Exception.class)
        .hasMessage(message);
    }

    @Test
    @DisplayName("Should support exception wrapping")
    void testExceptionWrapping() {
        // Given
        Exception original = new IllegalArgumentException("Invalid parameter");

        // When
        VTEAException wrapped = new VTEAException("Wrapper message", original);

        // Then
        assertThat(wrapped.getCause()).isEqualTo(original);
        assertThat(wrapped.getMessage()).isEqualTo("Wrapper message");
    }

    @Test
    @DisplayName("Should be serializable")
    void testSerialization() throws IOException, ClassNotFoundException {
        // Given
        String message = "Serialization test";
        VTEAException original = new VTEAException(message, new RuntimeException("cause"));

        // When - Serialize
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(original);
        oos.close();

        // When - Deserialize
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        ObjectInputStream ois = new ObjectInputStream(bais);
        VTEAException deserialized = (VTEAException) ois.readObject();
        ois.close();

        // Then
        assertThat(deserialized).isNotNull();
        assertThat(deserialized.getMessage()).isEqualTo(original.getMessage());
        assertThat(deserialized.getCause()).isNotNull();
    }

    @Test
    @DisplayName("Should have stack trace")
    void testStackTrace() {
        // Given/When
        VTEAException exception = new VTEAException("Test");

        // Then
        assertThat(exception.getStackTrace()).isNotNull();
        assertThat(exception.getStackTrace().length).isGreaterThan(0);
    }

    @Test
    @DisplayName("Should support null message")
    void testNullMessage() {
        // When
        VTEAException exception = new VTEAException((String) null);

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isNull();
    }

    @Test
    @DisplayName("Should support null cause")
    void testNullCause() {
        // When
        VTEAException exception = new VTEAException("Message", null);

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isEqualTo("Message");
        assertThat(exception.getCause()).isNull();
    }

    @Test
    @DisplayName("Should be catchable as Exception")
    void testCatchAsException() {
        // Then
        assertThatThrownBy(() -> {
            throw new VTEAException("Test");
        })
        .isInstanceOf(Exception.class);
    }

    @Test
    @DisplayName("Should be catchable as Throwable")
    void testCatchAsThrowable() {
        // Then
        assertThatThrownBy(() -> {
            throw new VTEAException("Test");
        })
        .isInstanceOf(Throwable.class);
    }
}
