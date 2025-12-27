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
package vtea.objects.measurements;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import vtea.BaseTest;

import java.util.ArrayList;
import java.util.Arrays;

import static org.assertj.core.api.Assertions.*;

/**
 * Test class for Count measurement
 */
@DisplayName("Count Measurement Tests")
class CountTest extends BaseTest {

    private Count count;
    private ArrayList<Number> values;

    @BeforeEach
    void setUp() {
        count = new Count();
        values = new ArrayList<>();
    }

    @Test
    @DisplayName("Should have correct metadata")
    void testMetadata() {
        assertThat(count.getName()).isEqualTo("Count");
        assertThat(count.getKey()).isEqualTo("Count");
        assertThat(count.getVersion()).isEqualTo("1.0");
        assertThat(count.getAuthor()).isEqualTo("Seth Winfree");
        assertThat(count.getComment()).isEqualTo("Calculate number of pixels");
        assertThat(count.getType()).isEqualTo("Shape");
    }

    @Test
    @DisplayName("Should count single value")
    void testCountSingleValue() {
        values.add(42);

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(1);
    }

    @Test
    @DisplayName("Should count multiple values")
    void testCountMultipleValues() {
        values.addAll(Arrays.asList(1, 2, 3, 4, 5));

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(5);
    }

    @Test
    @DisplayName("Should count ten values")
    void testCountTenValues() {
        for (int i = 0; i < 10; i++) {
            values.add(i);
        }

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(10);
    }

    @Test
    @DisplayName("Should count large number of values")
    void testCountLargeDataset() {
        for (int i = 0; i < 1000; i++) {
            values.add(i);
        }

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(1000);
    }

    @Test
    @DisplayName("Should count identical values correctly")
    void testCountIdenticalValues() {
        values.addAll(Arrays.asList(7, 7, 7, 7, 7, 7, 7));

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(7);
    }

    @Test
    @DisplayName("Should count regardless of value magnitude")
    void testCountWithDifferentMagnitudes() {
        values.addAll(Arrays.asList(0.001, 1000000, 0, -5000));

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(4);
    }

    @Test
    @DisplayName("Should count zeros")
    void testCountZeros() {
        values.addAll(Arrays.asList(0, 0, 0, 0));

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(4);
    }

    @Test
    @DisplayName("Should count negative values")
    void testCountNegativeValues() {
        values.addAll(Arrays.asList(-1, -2, -3, -4, -5));

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(5);
    }

    @Test
    @DisplayName("Should count mixed types of numbers")
    void testCountMixedTypes() {
        values.add(Integer.valueOf(10));
        values.add(Long.valueOf(20L));
        values.add(Float.valueOf(30.0f));
        values.add(Double.valueOf(40.0));

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(4);
    }

    @Test
    @DisplayName("Should be useful for pixel counting")
    void testPixelCountingUseCase() {
        // Simulate pixel intensity values - count should be independent of values
        values.addAll(Arrays.asList(255, 128, 64, 32, 16, 8, 4, 2, 1));

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(9);
    }

    @Test
    @DisplayName("Should count 100 random values")
    void testCountRandomValues() {
        for (int i = 0; i < 100; i++) {
            values.add(random.nextDouble() * 1000);
        }

        Number result = count.process(null, values);

        assertThat(result.intValue()).isEqualTo(100);
    }

    @Test
    @DisplayName("Should verify type is Shape not Intensity")
    void testTypeIsShape() {
        // Count is a shape measurement (number of pixels), not intensity
        assertThat(count.getType()).isEqualTo("Shape");
        assertThat(count.getType()).isNotEqualTo("Intensity");
    }
}
