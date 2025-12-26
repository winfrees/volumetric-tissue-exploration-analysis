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

import java.sql.SQLException;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for DatabaseException.
 *
 * @author VTEA Development Team
 */
@DisplayName("DatabaseException Tests")
class DatabaseExceptionTest extends BaseTest {

    @Test
    @DisplayName("Should extend VTEAException")
    void testInheritance() {
        // When
        DatabaseException exception = new DatabaseException();

        // Then
        assertThat(exception).isInstanceOf(VTEAException.class);
        assertThat(exception).isInstanceOf(Exception.class);
    }

    @Test
    @DisplayName("Should create with default constructor")
    void testDefaultConstructor() {
        // When
        DatabaseException exception = new DatabaseException();

        // Then
        assertThat(exception).isNotNull();
        assertThat(exception.getMessage()).isNull();
    }

    @Test
    @DisplayName("Should create with message")
    void testMessageConstructor() {
        // Given
        String message = "Database connection failed";

        // When
        DatabaseException exception = new DatabaseException(message);

        // Then
        assertThat(exception.getMessage()).isEqualTo(message);
    }

    @Test
    @DisplayName("Should create with message and cause")
    void testMessageAndCauseConstructor() {
        // Given
        String message = "Failed to insert measurement data";
        SQLException cause = new SQLException("Table does not exist");

        // When
        DatabaseException exception = new DatabaseException(message, cause);

        // Then
        assertThat(exception.getMessage()).isEqualTo(message);
        assertThat(exception.getCause()).isEqualTo(cause);
        assertThat(exception.getCause()).isInstanceOf(SQLException.class);
    }

    @Test
    @DisplayName("Should create with cause only")
    void testCauseConstructor() {
        // Given
        SQLException cause = new SQLException("Connection timeout");

        // When
        DatabaseException exception = new DatabaseException(cause);

        // Then
        assertThat(exception.getCause()).isEqualTo(cause);
    }

    @Test
    @DisplayName("Should be throwable for database errors")
    void testThrowable() {
        // Then
        assertThatThrownBy(() -> {
            throw new DatabaseException("Query execution failed");
        })
        .isInstanceOf(DatabaseException.class)
        .isInstanceOf(VTEAException.class)
        .hasMessageContaining("Query");
    }

    @Test
    @DisplayName("Should support real-world database error scenario")
    void testRealWorldScenario() {
        // Given
        String operation = "INSERT INTO measurements";
        String error = "Duplicate key violation";

        // When
        DatabaseException exception = new DatabaseException(
            operation + " failed: " + error
        );

        // Then
        assertThat(exception.getMessage())
            .contains("INSERT")
            .contains("Duplicate key");
    }

    @Test
    @DisplayName("Should wrap SQLException properly")
    void testSQLExceptionWrapping() {
        // Given
        SQLException sqlException = new SQLException(
            "Table 'MEASUREMENTS' not found",
            "42S02",  // SQL state
            1146      // Error code
        );

        // When
        DatabaseException exception = new DatabaseException(
            "Database query failed", sqlException
        );

        // Then
        assertThat(exception.getCause()).isInstanceOf(SQLException.class);
        SQLException wrappedSQL = (SQLException) exception.getCause();
        assertThat(wrappedSQL.getSQLState()).isEqualTo("42S02");
        assertThat(wrappedSQL.getErrorCode()).isEqualTo(1146);
    }

    @Test
    @DisplayName("Should be catchable as VTEAException")
    void testCatchAsVTEAException() {
        // Then
        assertThatThrownBy(() -> {
            throw new DatabaseException("Test");
        })
        .isInstanceOf(VTEAException.class);
    }

    @Test
    @DisplayName("Should support H2 database error scenarios")
    void testH2ErrorScenario() {
        // Given
        String h2Error = "H2 database connection pool exhausted";

        // When
        DatabaseException exception = new DatabaseException(h2Error);

        // Then
        assertThat(exception.getMessage()).contains("H2");
        assertThat(exception.getMessage()).contains("connection pool");
    }

    @Test
    @DisplayName("Should handle transaction rollback errors")
    void testTransactionError() {
        // Given
        SQLException transactionError = new SQLException("Transaction rollback failed");

        // When
        DatabaseException exception = new DatabaseException(
            "Failed to rollback transaction", transactionError
        );

        // Then
        assertThat(exception.getMessage()).contains("rollback");
        assertThat(exception.getCause().getMessage()).contains("rollback");
    }
}
