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

import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Test suite for all measurement classes
 *
 * Run this suite to execute all measurement tests together.
 *
 * Usage:
 *   mvn test -Dtest=AllMeasurementTests
 */
@Suite
@SuiteDisplayName("VTEA Measurement Tests Suite")
@SelectClasses({
    MeanTest.class,
    StandardDeviationTest.class,
    MinimumTest.class,
    MaximumTest.class,
    SumTest.class,
    CountTest.class,
    ThresholdMeanTest.class
})
public class AllMeasurementTests {
    // Suite configuration only - no implementation needed
}
