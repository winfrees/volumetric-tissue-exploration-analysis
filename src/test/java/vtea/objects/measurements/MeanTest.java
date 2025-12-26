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
 * Test class for Mean measurement
 */
@DisplayName("Mean Measurement Tests")
class MeanTest extends BaseTest {

    private Mean mean;
    private ArrayList<Number> values;

    @BeforeEach
    void setUp() {
        mean = new Mean();
        values = new ArrayList<>();
    }

    @Test
    @DisplayName("Should have correct metadata")
    void testMetadata() {
        assertThat(mean.getName()).isEqualTo("Mean");
        assertThat(mean.getKey()).isEqualTo("Mean");
        assertThat(mean.getVersion()).isEqualTo("1.0");
        assertThat(mean.getAuthor()).isEqualTo("Seth Winfree");
        assertThat(mean.getComment()).isEqualTo("Calculate mean");
        assertThat(mean.getType()).isEqualTo("Intensity");
    }

    @Test
    @DisplayName("Should calculate mean of positive integers correctly")
    void testMeanOfPositiveIntegers() {
        values.addAll(Arrays.asList(1, 2, 3, 4, 5));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(3.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate mean of positive doubles correctly")
    void testMeanOfPositiveDoubles() {
        values.addAll(Arrays.asList(1.5, 2.5, 3.5, 4.5));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(3.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate mean of mixed positive and negative values")
    void testMeanOfMixedValues() {
        values.addAll(Arrays.asList(-10, -5, 0, 5, 10));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(0.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate mean of single value")
    void testMeanOfSingleValue() {
        values.add(42);

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(42.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate mean of identical values")
    void testMeanOfIdenticalValues() {
        values.addAll(Arrays.asList(7, 7, 7, 7, 7));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(7.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate mean of large values")
    void testMeanOfLargeValues() {
        values.addAll(Arrays.asList(1000000, 2000000, 3000000));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(2000000.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate mean of very small values")
    void testMeanOfSmallValues() {
        values.addAll(Arrays.asList(0.001, 0.002, 0.003));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(0.002, within(0.0001));
    }

    @Test
    @DisplayName("Should calculate mean of 100 random values")
    void testMeanOfRandomValues() {
        double expectedSum = 0;
        for (int i = 0; i < 100; i++) {
            double value = random.nextDouble() * 100;
            values.add(value);
            expectedSum += value;
        }
        double expectedMean = expectedSum / 100;

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(expectedMean, within(DELTA));
    }

    @Test
    @DisplayName("process() method should delegate to getMean()")
    void testProcessMethod() {
        values.addAll(Arrays.asList(10, 20, 30, 40, 50));

        Number result = mean.process(null, values);

        assertThat(result.doubleValue()).isEqualTo(30.0, within(DELTA));
    }

    @Test
    @DisplayName("Should handle Integer, Long, Float, and Double types")
    void testMixedNumberTypes() {
        values.add(Integer.valueOf(10));
        values.add(Long.valueOf(20L));
        values.add(Float.valueOf(30.0f));
        values.add(Double.valueOf(40.0));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(25.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate mean with zero values included")
    void testMeanWithZeros() {
        values.addAll(Arrays.asList(0, 0, 10, 20));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(7.5, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate mean of negative values")
    void testMeanOfNegativeValues() {
        values.addAll(Arrays.asList(-5, -10, -15, -20));

        Number result = Mean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(-12.5, within(DELTA));
    }
}
