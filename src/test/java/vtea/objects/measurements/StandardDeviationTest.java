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
 * Test class for StandardDeviation measurement
 */
@DisplayName("StandardDeviation Measurement Tests")
class StandardDeviationTest extends BaseTest {

    private StandardDeviation standardDeviation;
    private ArrayList<Number> values;

    @BeforeEach
    void setUp() {
        standardDeviation = new StandardDeviation();
        values = new ArrayList<>();
    }

    @Test
    @DisplayName("Should have correct metadata")
    void testMetadata() {
        assertThat(standardDeviation.getName()).isEqualTo("StDev");
        assertThat(standardDeviation.getKey()).isEqualTo("SD");
        assertThat(standardDeviation.getVersion()).isEqualTo("1.0");
        assertThat(standardDeviation.getAuthor()).isEqualTo("Seth Winfree");
        assertThat(standardDeviation.getComment()).isEqualTo("Calculate standard deviation");
        assertThat(standardDeviation.getType()).isEqualTo("Intensity");
    }

    @Test
    @DisplayName("Should calculate standard deviation correctly for simple dataset")
    void testStandardDeviationSimple() {
        // Dataset: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, Sample StDev = 2.138
        values.addAll(Arrays.asList(2, 4, 4, 4, 5, 5, 7, 9));

        Number result = StandardDeviation.getStandardDeviation(values);

        assertThat(result.doubleValue()).isEqualTo(2.138, within(0.01));
    }

    @Test
    @DisplayName("Should calculate standard deviation of identical values as zero")
    void testStandardDeviationIdenticalValues() {
        values.addAll(Arrays.asList(5, 5, 5, 5, 5));

        Number result = StandardDeviation.getStandardDeviation(values);

        assertThat(result.doubleValue()).isEqualTo(0.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate standard deviation of two values")
    void testStandardDeviationTwoValues() {
        // Values: 1, 3
        // Mean = 2, Sample StDev = sqrt(2) ≈ 1.414
        values.addAll(Arrays.asList(1, 3));

        Number result = StandardDeviation.getStandardDeviation(values);

        assertThat(result.doubleValue()).isEqualTo(1.414, within(0.01));
    }

    @Test
    @DisplayName("Should calculate standard deviation of positive values")
    void testStandardDeviationPositiveValues() {
        // Values: 10, 12, 23, 23, 16, 23, 21, 16
        // Mean = 18, Sample StDev ≈ 5.237
        values.addAll(Arrays.asList(10, 12, 23, 23, 16, 23, 21, 16));

        Number result = StandardDeviation.getStandardDeviation(values);

        assertThat(result.doubleValue()).isEqualTo(5.237, within(0.01));
    }

    @Test
    @DisplayName("Should calculate standard deviation of negative values")
    void testStandardDeviationNegativeValues() {
        values.addAll(Arrays.asList(-5, -3, -1, -7, -9));

        Number result = StandardDeviation.getStandardDeviation(values);

        // Mean = -5, Sample StDev ≈ 3.162
        assertThat(result.doubleValue()).isEqualTo(3.162, within(0.01));
    }

    @Test
    @DisplayName("Should calculate standard deviation of mixed positive and negative values")
    void testStandardDeviationMixedValues() {
        values.addAll(Arrays.asList(-10, -5, 0, 5, 10));

        Number result = StandardDeviation.getStandardDeviation(values);

        // Mean = 0, Sample StDev ≈ 7.906
        assertThat(result.doubleValue()).isEqualTo(7.906, within(0.01));
    }

    @Test
    @DisplayName("Should calculate standard deviation of decimal values")
    void testStandardDeviationDecimalValues() {
        values.addAll(Arrays.asList(1.5, 2.5, 3.5, 4.5, 5.5));

        Number result = StandardDeviation.getStandardDeviation(values);

        // Mean = 3.5, Sample StDev ≈ 1.581
        assertThat(result.doubleValue()).isEqualTo(1.581, within(0.01));
    }

    @Test
    @DisplayName("Should calculate standard deviation with large spread")
    void testStandardDeviationLargeSpread() {
        values.addAll(Arrays.asList(1, 100, 1, 100, 1, 100));

        Number result = StandardDeviation.getStandardDeviation(values);

        // Mean = 50.5, Sample StDev ≈ 54.01
        assertThat(result.doubleValue()).isEqualTo(54.01, within(0.1));
    }

    @Test
    @DisplayName("Should calculate standard deviation with small spread")
    void testStandardDeviationSmallSpread() {
        values.addAll(Arrays.asList(100.0, 100.1, 100.2, 100.1, 100.0));

        Number result = StandardDeviation.getStandardDeviation(values);

        // Mean = 100.08, Sample StDev ≈ 0.084
        assertThat(result.doubleValue()).isEqualTo(0.084, within(0.01));
    }

    @Test
    @DisplayName("process() method should delegate to getStandardDeviation()")
    void testProcessMethod() {
        values.addAll(Arrays.asList(2, 4, 6, 8, 10));

        Number result = standardDeviation.process(null, values);

        // Mean = 6, Sample StDev ≈ 3.162
        assertThat(result.doubleValue()).isEqualTo(3.162, within(0.01));
    }

    @Test
    @DisplayName("Should handle Integer, Long, Float, and Double types")
    void testMixedNumberTypes() {
        values.add(Integer.valueOf(10));
        values.add(Long.valueOf(20L));
        values.add(Float.valueOf(30.0f));
        values.add(Double.valueOf(40.0));

        Number result = StandardDeviation.getStandardDeviation(values);

        // Mean = 25, Sample StDev ≈ 12.91
        assertThat(result.doubleValue()).isEqualTo(12.91, within(0.1));
    }

    @Test
    @DisplayName("Should calculate standard deviation with zeros")
    void testStandardDeviationWithZeros() {
        values.addAll(Arrays.asList(0, 0, 10, 20));

        Number result = StandardDeviation.getStandardDeviation(values);

        // Mean = 7.5, Sample StDev ≈ 9.574
        assertThat(result.doubleValue()).isEqualTo(9.574, within(0.01));
    }

    @Test
    @DisplayName("Should calculate standard deviation of large dataset")
    void testStandardDeviationLargeDataset() {
        // Generate 100 values from 1 to 100
        for (int i = 1; i <= 100; i++) {
            values.add(i);
        }

        Number result = StandardDeviation.getStandardDeviation(values);

        // For 1 to 100, sample StDev ≈ 29.01
        assertThat(result.doubleValue()).isEqualTo(29.01, within(0.1));
    }

    @Test
    @DisplayName("Should use sample standard deviation formula (n-1 denominator)")
    void testSampleStandardDeviationFormula() {
        // Verify it uses n-1 (sample) not n (population)
        // For values 1, 2, 3:
        // Population StDev = sqrt(2/3) ≈ 0.816
        // Sample StDev = sqrt(2/2) = 1.0
        values.addAll(Arrays.asList(1, 2, 3));

        Number result = StandardDeviation.getStandardDeviation(values);

        assertThat(result.doubleValue()).isEqualTo(1.0, within(DELTA));
    }
}
