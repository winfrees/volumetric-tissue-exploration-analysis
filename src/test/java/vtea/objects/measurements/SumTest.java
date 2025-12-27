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
 * Test class for Sum measurement
 */
@DisplayName("Sum Measurement Tests")
class SumTest extends BaseTest {

    private Sum sum;
    private ArrayList<Number> values;

    @BeforeEach
    void setUp() {
        sum = new Sum();
        values = new ArrayList<>();
    }

    @Test
    @DisplayName("Should have correct metadata")
    void testMetadata() {
        assertThat(sum.getName()).isEqualTo("Sum");
        assertThat(sum.getKey()).isEqualTo("Sum");
        assertThat(sum.getVersion()).isEqualTo("1.0");
        assertThat(sum.getAuthor()).isEqualTo("Seth Winfree");
        assertThat(sum.getComment()).isEqualTo("Calculate sum or integrated density");
        assertThat(sum.getType()).isEqualTo("Intensity");
    }

    @Test
    @DisplayName("Should calculate sum of positive integers")
    void testSumOfPositiveIntegers() {
        values.addAll(Arrays.asList(1, 2, 3, 4, 5));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(15.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of positive doubles")
    void testSumOfPositiveDoubles() {
        values.addAll(Arrays.asList(1.5, 2.5, 3.5, 4.5));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(12.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of negative values")
    void testSumOfNegativeValues() {
        values.addAll(Arrays.asList(-1, -2, -3, -4, -5));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(-15.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of mixed positive and negative values")
    void testSumOfMixedValues() {
        values.addAll(Arrays.asList(-10, -5, 5, 10));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(0.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of single value")
    void testSumOfSingleValue() {
        values.add(42);

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(42.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of identical values")
    void testSumOfIdenticalValues() {
        values.addAll(Arrays.asList(7, 7, 7, 7, 7));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(35.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum with zeros")
    void testSumWithZeros() {
        values.addAll(Arrays.asList(0, 5, 0, 10, 0));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(15.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of all zeros")
    void testSumOfAllZeros() {
        values.addAll(Arrays.asList(0, 0, 0, 0));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(0.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of large values")
    void testSumOfLargeValues() {
        values.addAll(Arrays.asList(1000000, 2000000, 3000000));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(6000000.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of very small values")
    void testSumOfSmallValues() {
        values.addAll(Arrays.asList(0.001, 0.002, 0.003, 0.004));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(0.010, within(0.0001));
    }

    @Test
    @DisplayName("Should handle Integer, Long, Float, and Double types")
    void testMixedNumberTypes() {
        values.add(Integer.valueOf(10));
        values.add(Long.valueOf(20L));
        values.add(Float.valueOf(30.0f));
        values.add(Double.valueOf(40.0));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(100.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of 100 values")
    void testSumOfManyValues() {
        double expectedSum = 0;
        for (int i = 1; i <= 100; i++) {
            values.add(i);
            expectedSum += i;
        }

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(expectedSum, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum for integrated density use case")
    void testIntegratedDensity() {
        // Simulating pixel intensity values for integrated density calculation
        values.addAll(Arrays.asList(128, 255, 64, 192, 32));

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(671.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate sum of random values correctly")
    void testSumOfRandomValues() {
        double expectedSum = 0;
        for (int i = 0; i < 50; i++) {
            double value = random.nextDouble() * 100;
            values.add(value);
            expectedSum += value;
        }

        Number result = sum.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(expectedSum, within(DELTA));
    }
}
