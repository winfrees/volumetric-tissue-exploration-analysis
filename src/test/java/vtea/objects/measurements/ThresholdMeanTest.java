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
 * Test class for ThresholdMean measurement
 *
 * ThresholdMean calculates the mean of values above a cutoff threshold.
 * Cutoff = max - ((max - min) / 4), which captures the top 25% of the range.
 */
@DisplayName("ThresholdMean Measurement Tests")
class ThresholdMeanTest extends BaseTest {

    private ThresholdMean thresholdMean;
    private ArrayList<Number> values;

    @BeforeEach
    void setUp() {
        thresholdMean = new ThresholdMean();
        values = new ArrayList<>();
    }

    @Test
    @DisplayName("Should have correct metadata")
    void testMetadata() {
        assertThat(thresholdMean.getName()).isEqualTo("Mean Threshold");
        assertThat(thresholdMean.getKey()).isEqualTo("MeanThresh");
        assertThat(thresholdMean.getVersion()).isEqualTo("1.0");
        assertThat(thresholdMean.getAuthor()).isEqualTo("Seth Winfree");
        assertThat(thresholdMean.getComment()).isEqualTo("Calculate thresholded mean, mean of top 25% of values");
        assertThat(thresholdMean.getType()).isEqualTo("Intensity");
    }

    @Test
    @DisplayName("Should calculate threshold mean for simple range 0-100")
    void testThresholdMeanSimpleRange() {
        // Range 0-100, cutoff = 100 - ((100-0)/4) = 100 - 25 = 75
        // Values >= 75: 80, 90, 100
        values.addAll(Arrays.asList(0, 20, 40, 60, 80, 90, 100));

        Number result = ThresholdMean.getMean(values);

        // Mean of 80, 90, 100 = 90
        assertThat(result.doubleValue()).isEqualTo(90.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate threshold mean for range 0-10")
    void testThresholdMeanSmallRange() {
        // Range 0-10, cutoff = 10 - ((10-0)/4) = 10 - 2.5 = 7.5
        // Values >= 7.5: 8, 9, 10
        values.addAll(Arrays.asList(0, 2, 4, 6, 8, 9, 10));

        Number result = ThresholdMean.getMean(values);

        // Mean of 8, 9, 10 = 9
        assertThat(result.doubleValue()).isEqualTo(9.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate threshold mean for identical values")
    void testThresholdMeanIdenticalValues() {
        // All values are 50, cutoff = 50 - ((50-50)/4) = 50
        // All values >= 50 should be included
        values.addAll(Arrays.asList(50, 50, 50, 50, 50));

        Number result = ThresholdMean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(50.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate threshold mean for two distinct values")
    void testThresholdMeanTwoValues() {
        // Range 10-20, cutoff = 20 - ((20-10)/4) = 20 - 2.5 = 17.5
        // Only value 20 >= 17.5
        values.addAll(Arrays.asList(10, 20));

        Number result = ThresholdMean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(20.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate threshold mean with negative values")
    void testThresholdMeanNegativeValues() {
        // Range -100 to 0, cutoff = 0 - ((0-(-100))/4) = 0 - 25 = -25
        // Values >= -25: -20, -10, 0
        values.addAll(Arrays.asList(-100, -60, -40, -20, -10, 0));

        Number result = ThresholdMean.getMean(values);

        // Mean of -20, -10, 0 = -10
        assertThat(result.doubleValue()).isEqualTo(-10.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate threshold mean for mixed positive and negative values")
    void testThresholdMeanMixedValues() {
        // Range -50 to 50, cutoff = 50 - ((50-(-50))/4) = 50 - 25 = 25
        // Values >= 25: 30, 40, 50
        values.addAll(Arrays.asList(-50, -25, 0, 10, 30, 40, 50));

        Number result = ThresholdMean.getMean(values);

        // Mean of 30, 40, 50 = 40
        assertThat(result.doubleValue()).isEqualTo(40.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate threshold mean for single value")
    void testThresholdMeanSingleValue() {
        // Single value, cutoff = 42 - ((42-42)/4) = 42
        // The value 42 >= 42 is included
        values.add(42);

        Number result = ThresholdMean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(42.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate threshold mean for decimal values")
    void testThresholdMeanDecimalValues() {
        // Range 1.0-5.0, cutoff = 5.0 - ((5.0-1.0)/4) = 5.0 - 1.0 = 4.0
        // Values >= 4.0: 4.0, 4.5, 5.0
        values.addAll(Arrays.asList(1.0, 2.0, 3.0, 4.0, 4.5, 5.0));

        Number result = ThresholdMean.getMean(values);

        // Mean of 4.0, 4.5, 5.0 = 4.5
        assertThat(result.doubleValue()).isEqualTo(4.5, within(DELTA));
    }

    @Test
    @DisplayName("Should handle pixel intensity values (0-255)")
    void testThresholdMeanPixelIntensities() {
        // Range 0-255, cutoff = 255 - ((255-0)/4) = 255 - 63.75 = 191.25
        // Values >= 191.25: 200, 220, 240, 255
        values.addAll(Arrays.asList(0, 50, 100, 150, 200, 220, 240, 255));

        Number result = ThresholdMean.getMean(values);

        // Mean of 200, 220, 240, 255 = 228.75
        assertThat(result.doubleValue()).isEqualTo(228.75, within(DELTA));
    }

    @Test
    @DisplayName("Should include values at exactly the cutoff threshold")
    void testThresholdMeanIncludesExactCutoff() {
        // Range 0-100, cutoff = 75
        // Test that value exactly at 75 is included
        values.addAll(Arrays.asList(0, 25, 50, 75, 100));

        Number result = ThresholdMean.getMean(values);

        // Mean of 75, 100 = 87.5
        assertThat(result.doubleValue()).isEqualTo(87.5, within(DELTA));
    }

    @Test
    @DisplayName("process() method should delegate to getMean()")
    void testProcessMethod() {
        // Range 0-100, cutoff = 75
        values.addAll(Arrays.asList(0, 20, 40, 60, 80, 90, 100));

        Number result = thresholdMean.process(null, values);

        // Mean of 80, 90, 100 = 90
        assertThat(result.doubleValue()).isEqualTo(90.0, within(DELTA));
    }

    @Test
    @DisplayName("Should handle Integer, Long, Float, and Double types")
    void testMixedNumberTypes() {
        // Range 10-40, cutoff = 40 - ((40-10)/4) = 40 - 7.5 = 32.5
        // Values >= 32.5: 40.0
        values.add(Integer.valueOf(10));
        values.add(Long.valueOf(20L));
        values.add(Float.valueOf(30.0f));
        values.add(Double.valueOf(40.0));

        Number result = ThresholdMean.getMean(values);

        assertThat(result.doubleValue()).isEqualTo(40.0, within(DELTA));
    }

    @Test
    @DisplayName("Should calculate threshold mean with many values in top 25%")
    void testThresholdMeanManyTopValues() {
        // Range 0-100, cutoff = 75
        // Many values >= 75
        values.addAll(Arrays.asList(0, 10, 20, 30, 75, 80, 85, 90, 95, 100));

        Number result = ThresholdMean.getMean(values);

        // Mean of 75, 80, 85, 90, 95, 100 = 87.5
        assertThat(result.doubleValue()).isEqualTo(87.5, within(DELTA));
    }

    @Test
    @DisplayName("Should handle large range of values")
    void testThresholdMeanLargeRange() {
        // Range 0-1000000, cutoff = 1000000 - 250000 = 750000
        values.addAll(Arrays.asList(0, 250000, 500000, 750000, 1000000));

        Number result = ThresholdMean.getMean(values);

        // Mean of 750000, 1000000 = 875000
        assertThat(result.doubleValue()).isEqualTo(875000.0, within(DELTA));
    }

    @Test
    @DisplayName("Should be useful for finding bright pixel mean in microscopy")
    void testMicroscopyUseCase() {
        // Simulate microscopy image with background (low values) and bright features (high values)
        // Background: 10-20, Features: 180-220
        values.addAll(Arrays.asList(10, 12, 15, 18, 20, 180, 200, 220));

        // Range 10-220, cutoff = 220 - ((220-10)/4) = 220 - 52.5 = 167.5
        // Values >= 167.5: 180, 200, 220
        Number result = ThresholdMean.getMean(values);

        // Mean of bright pixels: 200
        assertThat(result.doubleValue()).isEqualTo(200.0, within(DELTA));
    }
}
