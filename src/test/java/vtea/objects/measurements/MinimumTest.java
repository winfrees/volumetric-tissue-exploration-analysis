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
 * Test class for Minimum measurement
 */
@DisplayName("Minimum Measurement Tests")
class MinimumTest extends BaseTest {

    private Minimum minimum;
    private ArrayList<Number> values;

    @BeforeEach
    void setUp() {
        minimum = new Minimum();
        values = new ArrayList<>();
    }

    @Test
    @DisplayName("Should have correct metadata")
    void testMetadata() {
        assertThat(minimum.getName()).isEqualTo("Min");
        assertThat(minimum.getKey()).isEqualTo("Min");
        assertThat(minimum.getVersion()).isEqualTo("1.0");
        assertThat(minimum.getAuthor()).isEqualTo("Seth Winfree");
        assertThat(minimum.getComment()).isEqualTo("Calculate minimum");
        assertThat(minimum.getType()).isEqualTo("Intensity");
    }

    @Test
    @DisplayName("Should find minimum of positive integers")
    void testMinimumOfPositiveIntegers() {
        values.addAll(Arrays.asList(5, 3, 8, 1, 9));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(1.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum of negative integers")
    void testMinimumOfNegativeIntegers() {
        values.addAll(Arrays.asList(-5, -3, -8, -1, -9));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(-9.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum of mixed positive and negative values")
    void testMinimumOfMixedValues() {
        values.addAll(Arrays.asList(10, -5, 0, 15, -20));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(-20.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum of single value")
    void testMinimumOfSingleValue() {
        values.add(42);

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(42.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum of identical values")
    void testMinimumOfIdenticalValues() {
        values.addAll(Arrays.asList(7, 7, 7, 7, 7));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(7.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum when it's the first element")
    void testMinimumAsFirstElement() {
        values.addAll(Arrays.asList(1, 100, 50, 75, 25));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(1.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum when it's the last element")
    void testMinimumAsLastElement() {
        values.addAll(Arrays.asList(100, 50, 75, 25, 1));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(1.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum with zero value")
    void testMinimumWithZero() {
        values.addAll(Arrays.asList(10, 5, 0, 15, 20));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(0.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum of decimal values")
    void testMinimumOfDecimalValues() {
        values.addAll(Arrays.asList(1.5, 0.3, 2.7, 0.1, 3.9));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(0.1, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum of very small values")
    void testMinimumOfVerySmallValues() {
        values.addAll(Arrays.asList(0.001, 0.0001, 0.01, 0.1));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(0.0001, within(0.00001));
    }

    @Test
    @DisplayName("Should find minimum of very large values")
    void testMinimumOfVeryLargeValues() {
        values.addAll(Arrays.asList(1000000, 500000, 2000000, 750000));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(500000.0, within(DELTA));
    }

    @Test
    @DisplayName("process() method should delegate to getMinimum()")
    void testProcessMethod() {
        values.addAll(Arrays.asList(10, 20, 5, 30, 40));

        Number result = minimum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(5.0, within(DELTA));
    }

    @Test
    @DisplayName("Should handle Integer, Long, Float, and Double types")
    void testMixedNumberTypes() {
        values.add(Integer.valueOf(10));
        values.add(Long.valueOf(5L));
        values.add(Float.valueOf(20.0f));
        values.add(Double.valueOf(3.0));

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(3.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find minimum in large dataset")
    void testMinimumInLargeDataset() {
        // Add 100 random values and track the minimum
        double expectedMin = Double.MAX_VALUE;
        for (int i = 0; i < 100; i++) {
            double value = random.nextDouble() * 1000;
            values.add(value);
            expectedMin = Math.min(expectedMin, value);
        }

        Number result = Minimum.getMinimum(values);

        assertThat(result.doubleValue()).isEqualTo(expectedMin, within(DELTA));
    }
}
