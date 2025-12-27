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

import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/**
 * Test suite that runs all exception tests together.
 *
 * This suite can be run to verify all exception classes in one execution.
 * Useful for quick verification of exception handling infrastructure.
 *
 * Run with: mvn test -Dtest=AllExceptionTests
 *
 * @author VTEA Development Team
 */
@Suite
@SuiteDisplayName("VTEA Exception Tests Suite")
@SelectClasses({
    VTEAExceptionTest.class,
    ImageProcessingExceptionTest.class,
    SegmentationExceptionTest.class,
    DatabaseExceptionTest.class,
    ClusteringExceptionTest.class
})
public class AllExceptionTests {
    // This class remains empty - it's used only as a holder for the above annotations
}
