# VTEA Test Suite

This directory contains the test suite for the VTEA (Volumetric Tissue Exploration and Analysis) project.

## Test Framework

- **JUnit 5 (Jupiter)** - Main testing framework
- **Mockito 5.x** - Mocking framework for dependencies
- **AssertJ** - Fluent assertions library
- **Hamcrest** - Additional matchers

## Running Tests

```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=MicroObjectTest

# Run tests with coverage report
mvn clean test jacoco:report

# View coverage report
open target/site/jacoco/index.html
```

## Test Organization

Tests should mirror the main source structure:

```
src/test/java/
├── vtea/
│   ├── clustering/          # Clustering algorithm tests
│   ├── exceptions/          # Exception class tests
│   ├── jdbc/                # Database tests
│   ├── objects/             # Segmentation tests
│   ├── processor/           # Processor tests
│   └── ...
├── vteaobjects/             # Core data model tests
└── BaseTest.java            # Base test utilities
```

## Test Naming Conventions

- Test classes: `<ClassName>Test.java` (e.g., `MicroObjectTest.java`)
- Test methods: `test<MethodName>_<Scenario>()` or descriptive names
- Use `@DisplayName` for readable test names

## Writing Tests

### Example Test Class

```java
package vtea.objects;

import org.junit.jupiter.api.*;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MicroObjectTest {

    private MicroObject microObject;

    @BeforeEach
    void setUp() {
        microObject = new MicroObject();
    }

    @Test
    @DisplayName("Should calculate centroid correctly for single pixel")
    void testCentroidCalculation_singlePixel() {
        // Given
        ArrayList<int[]> pixels = createSinglePixel(5, 5, 0);

        // When
        MicroObject obj = new MicroObject(pixels, 0, mockStack, 1);

        // Then
        assertThat(obj.getCentroidX()).isCloseTo(5.0f, within(0.01f));
        assertThat(obj.getCentroidY()).isCloseTo(5.0f, within(0.01f));
    }
}
```

## Test Categories

### Unit Tests
Test individual classes and methods in isolation.
- Fast execution (< 1 second per test)
- Use mocks for dependencies
- Focus on logic and algorithms

### Integration Tests
Test multiple components working together.
- May be slower
- Use real dependencies where appropriate
- Test workflows and pipelines

## Code Coverage

Target: **50% line coverage**

Run coverage report:
```bash
mvn clean test jacoco:report
```

View report at: `target/site/jacoco/index.html`

## Best Practices

1. **One Assert Per Test** - Makes failures clear
2. **Descriptive Names** - Use @DisplayName or clear method names
3. **AAA Pattern** - Arrange, Act, Assert
4. **Fast Tests** - Mock slow operations (I/O, database)
5. **Independent Tests** - Each test should run in isolation
6. **Test Data Builders** - Create reusable test data factories
7. **Don't Test Libraries** - Trust SMILE, ImageJ; test your usage

## Test Data

Test resources go in `src/test/resources/`:
- Small synthetic images for testing
- Sample data files
- Test configuration files

## Continuous Integration

Tests run automatically on:
- Every commit (via GitHub Actions)
- Pull requests
- Nightly builds

## Coverage Goals by Phase

- Phase 1 (Week 1): 10% coverage
- Phase 2 (Week 2): 25% coverage
- Phase 3 (Week 3): 40% coverage
- Phase 4 (Week 4): 50% coverage

## Help & Resources

- [JUnit 5 User Guide](https://junit.org/junit5/docs/current/user-guide/)
- [Mockito Documentation](https://javadoc.io/doc/org.mockito/mockito-core/latest/org/mockito/Mockito.html)
- [AssertJ Documentation](https://assertj.github.io/doc/)
- See TESTING_ROADMAP.md for detailed task breakdown
- See TESTING_QUICK_START.md for getting started guide
