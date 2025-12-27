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
package vtea;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

/**
 * Smoke tests for VTEA core functionality.
 *
 * These tests verify that basic components load and initialize correctly.
 * They should be fast and run before any other tests.
 *
 * @author VTEA Development Team
 */
@DisplayName("VTEA Smoke Tests")
class VTEASmokeTest extends BaseTest {

    @Test
    @DisplayName("Main VTEA class should load successfully")
    void testMainClassLoads() {
        // Given/When
        Class<_vtea> vteaClass = _vtea.class;

        // Then
        assertThat(vteaClass).isNotNull();
        assertThat(vteaClass.getName()).isEqualTo("vtea._vtea");
    }

    @Test
    @DisplayName("VTEA version constant should be set")
    void testVersionIsSet() {
        // Given/When
        String version = _vtea.VERSION;

        // Then
        assertThat(version)
                .isNotNull()
                .isNotEmpty()
                .matches("\\d+\\.\\d+\\.\\d+"); // Matches semantic versioning like 1.2.3
    }

    @Test
    @DisplayName("VTEA should have correct version number")
    void testVersionNumber() {
        // Given/When
        String version = _vtea.VERSION;

        // Then
        assertThat(version).isEqualTo("1.2.3");
    }

    @Test
    @DisplayName("Default directories should be configured")
    void testDefaultDirectoriesConfigured() {
        // Given/When
        String tempDir = _vtea.TEMP_DIRECTORY;
        String plotDir = _vtea.PLOT_DIRECTORY;
        String dbDir = _vtea.DATABASE_DIRECTORY;

        // Then
        assertThat(tempDir).isNotNull().isNotEmpty();
        assertThat(plotDir).isNotNull().isNotEmpty();
        assertThat(dbDir).isNotNull().isNotEmpty();
    }

    @Test
    @DisplayName("Database constants should be set")
    void testDatabaseConstantsSet() {
        // Given/When
        String dbName = _vtea.H2_DATABASE;
        String measurementsTable = _vtea.H2_MEASUREMENTS_TABLE;
        String objectsTable = _vtea.H2_OBJECT_TABLE;

        // Then
        assertThat(dbName).isEqualTo("VTEADB");
        assertThat(measurementsTable).isEqualTo("MEASUREMENTS");
        assertThat(objectsTable).isEqualTo("OBJECTS");
    }

    @Test
    @DisplayName("Feature types should be defined")
    void testFeatureTypesDefined() {
        // Given/When
        String[] featureTypes = _vtea.FEATURETYPE;

        // Then
        assertThat(featureTypes)
                .isNotNull()
                .hasSize(3)
                .contains("Cluster", "Reduction", "Other");
    }

    @Test
    @DisplayName("Measurement types should be defined")
    void testMeasurementTypesDefined() {
        // Given/When
        String[] measurementTypes = _vtea.MEASUREMENTTYPE;

        // Then
        assertThat(measurementTypes)
                .isNotNull()
                .hasSize(4)
                .contains("Intensity", "Shape", "Texture", "Relationship");
    }

    @Test
    @DisplayName("Map maker options should be defined")
    void testMapMakerOptionsDefined() {
        // Given/When
        String[] mapMakerOptions = _vtea.MAPMAKEROPTIONS;

        // Then
        assertThat(mapMakerOptions)
                .isNotNull()
                .hasSize(2)
                .contains("Feature", "Gate");
    }

    @Test
    @DisplayName("UI color constants should be set")
    void testUIColorsSet() {
        // Given/When
        java.awt.Color background = _vtea.BACKGROUND;
        java.awt.Color buttonBackground = _vtea.BUTTONBACKGROUND;
        java.awt.Color activeText = _vtea.ACTIVETEXT;
        java.awt.Color inactiveText = _vtea.INACTIVETEXT;

        // Then
        assertThat(background).isNotNull();
        assertThat(buttonBackground).isNotNull();
        assertThat(activeText).isNotNull();
        assertThat(inactiveText).isNotNull();
    }

    @Test
    @DisplayName("UI dimension constants should be set")
    void testUIDimensionsSet() {
        // Given/When
        java.awt.Dimension smallButtonSize = _vtea.SMALLBUTTONSIZE;
        java.awt.Dimension blockSetup = _vtea.BLOCKSETUP;

        // Then
        assertThat(smallButtonSize).isNotNull();
        assertThat(blockSetup).isNotNull();
        assertThat(smallButtonSize.width).isPositive();
        assertThat(smallButtonSize.height).isPositive();
        assertThat(blockSetup.width).isPositive();
        assertThat(blockSetup.height).isPositive();
    }

    @Test
    @DisplayName("Memory utility methods should work")
    void testMemoryUtilities() {
        // Given/When
        long availableMemory = _vtea.getAvailableMemory();
        long possibleThreads = _vtea.getPossibleThreads(1024 * 1024); // 1MB stack size

        // Then
        assertThat(availableMemory).isPositive();
        assertThat(possibleThreads).isPositive();
    }

    @Test
    @DisplayName("Database should be configured for in-memory by default")
    void testDatabaseInMemoryDefault() {
        // Given/When
        boolean inMemory = _vtea.DATABASE_IN_RAM;

        // Then
        assertThat(inMemory).isTrue();
    }
}
