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
 * Test class for Maximum measurement
 */
@DisplayName("Maximum Measurement Tests")
class MaximumTest extends BaseTest {

    private Maximum maximum;
    private ArrayList<Number> values;

    @BeforeEach
    void setUp() {
        maximum = new Maximum();
        values = new ArrayList<>();
    }

    @Test
    @DisplayName("Should have correct metadata")
    void testMetadata() {
        assertThat(maximum.getName()).isEqualTo("Max");
        assertThat(maximum.getKey()).isEqualTo("Max");
        assertThat(maximum.getVersion()).isEqualTo("1.0");
        assertThat(maximum.getAuthor()).isEqualTo("Seth Winfree");
        assertThat(maximum.getComment()).isEqualTo("Calculate maximum value");
        assertThat(maximum.getType()).isEqualTo("Intensity");
    }

    @Test
    @DisplayName("Should find maximum of positive integers")
    void testMaximumOfPositiveIntegers() {
        values.addAll(Arrays.asList(5, 3, 8, 1, 9));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(9.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum of negative integers")
    void testMaximumOfNegativeIntegers() {
        values.addAll(Arrays.asList(-5, -3, -8, -1, -9));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(-1.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum of mixed positive and negative values")
    void testMaximumOfMixedValues() {
        values.addAll(Arrays.asList(-10, 5, 0, -15, 20));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(20.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum of single value")
    void testMaximumOfSingleValue() {
        values.add(42);

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(42.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum of identical values")
    void testMaximumOfIdenticalValues() {
        values.addAll(Arrays.asList(7, 7, 7, 7, 7));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(7.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum when it's the first element")
    void testMaximumAsFirstElement() {
        values.addAll(Arrays.asList(100, 50, 75, 25, 1));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(100.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum when it's the last element")
    void testMaximumAsLastElement() {
        values.addAll(Arrays.asList(50, 75, 25, 1, 100));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(100.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum with zero value")
    void testMaximumWithZero() {
        values.addAll(Arrays.asList(-10, -5, 0, -15, -20));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(0.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum of decimal values")
    void testMaximumOfDecimalValues() {
        values.addAll(Arrays.asList(1.5, 3.9, 2.7, 0.1, 3.8));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(3.9, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum of very small values")
    void testMaximumOfVerySmallValues() {
        values.addAll(Arrays.asList(0.001, 0.0001, 0.01, 0.1));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(0.1, within(0.00001));
    }

    @Test
    @DisplayName("Should find maximum of very large values")
    void testMaximumOfVeryLargeValues() {
        values.addAll(Arrays.asList(1000000, 2000000, 500000, 750000));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(2000000.0, within(DELTA));
    }

    @Test
    @DisplayName("process() method should delegate to getMaximum()")
    void testProcessMethod() {
        values.addAll(Arrays.asList(10, 20, 5, 30, 40));

        Number result = maximum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(40.0, within(DELTA));
    }

    @Test
    @DisplayName("Should handle Integer, Long, Float, and Double types")
    void testMixedNumberTypes() {
        values.add(Integer.valueOf(10));
        values.add(Long.valueOf(50L));
        values.add(Float.valueOf(20.0f));
        values.add(Double.valueOf(30.0));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(50.0, within(DELTA));
    }

    @Test
    @DisplayName("Should find maximum in large dataset")
    void testMaximumInLargeDataset() {
        // Add 100 random values and track the maximum
        double expectedMax = Double.MIN_VALUE;
        for (int i = 0; i < 100; i++) {
            double value = random.nextDouble() * 1000;
            values.add(value);
            expectedMax = Math.max(expectedMax, value);
        }

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(expectedMax, within(DELTA));
    }

    @Test
    @DisplayName("Should handle all positive values starting from zero")
    void testAllPositiveFromZero() {
        values.addAll(Arrays.asList(0, 1, 2, 3, 4, 5));

        Number result = Maximum.getMaximum(values);

        assertThat(result.doubleValue()).isEqualTo(5.0, within(DELTA));
    }
}
