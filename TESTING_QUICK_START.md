# VTEA Testing - Quick Start Guide

**Goal:** Get from 0% to 50% test coverage in 3-4 weeks

---

## 📅 Week-by-Week Plan

### Week 1: Foundation (10-14 hours)
**Coverage: 0% → 10%**

| Task | Time | What You'll Test |
|------|------|------------------|
| 1.1 Set up framework | 2-3h | JUnit 5 + Mockito + AssertJ |
| 1.2 Base classes | 1-2h | Test utilities & helpers |
| 1.3 Smoke test | 30m | Main class loads correctly |
| 3.1 Custom exceptions | 2-3h | All 5 exception classes |
| 2.1 MicroObject | 4-6h | Core data model |

**Deliverable:** `mvn test` works, first 15+ tests passing ✅

---

### Week 2: Core Algorithms (14-18 hours)
**Coverage: 10% → 25%**

| Task | Time | What You'll Test |
|------|------|------------------|
| 5.1 K-Means | 4-5h | Clustering algorithm |
| 6.1 Simple segmentation | 3-4h | Threshold detection |
| 7.1 Database | 4-5h | H2 operations |
| 4.3 Measurements | 3-4h | Mean, stddev, min/max |

**Deliverable:** Critical algorithms validated ✅

---

### Week 3: Processors (14-18 hours)
**Coverage: 25% → 40%**

| Task | Time | What You'll Test |
|------|------|------------------|
| 8.1 SegmentationProcessor | 4-5h | Async processing |
| 8.2 FeatureProcessor | 3-4h | Clustering pipeline |
| 9.1 E2E segmentation | 4-5h | Full workflow |
| 4.1 Image processing | 3-4h | Filters & transforms |

**Deliverable:** Processor layer solid, integration tests ✅

---

### Week 4: Polish (11-14 hours)
**Coverage: 40% → 50%**

| Task | Time | What You'll Test |
|------|------|------------------|
| 5.2 Hierarchical clustering | 4-5h | Ward, Single, Complete |
| 6.3 3D segmentation | 5-6h | LayerCake algorithm |
| 2.2 MicroNeighborhoodObject | 2-3h | Spatial relationships |

**Deliverable:** 50% coverage, production-ready ✅

---

## 🎯 First Day Tasks (4-6 hours)

**Morning:**
1. Add test dependencies to `pom.xml` (30 min)
2. Create `src/test/java` structure (15 min)
3. Configure Maven Surefire plugin (15 min)
4. Run `mvn test` - should pass with 0 tests (15 min)

**Afternoon:**
5. Create `BaseTest.java` with utilities (1 hour)
6. Create `VTEASmokeTest.java` (30 min)
7. Write first test - assert VERSION constant (15 min)
8. Make it pass - run `mvn test` ✅ (15 min)

**End of Day:**
- ✅ Test infrastructure working
- ✅ First test passing
- ✅ Coverage report generating

---

## 📊 Task Breakdown by Complexity

### 🟢 Easy Tasks (2-3 hours each)
- Task 1.3: Smoke test
- Task 3.1: Exception tests
- Task 4.3: Measurement tests
- Task 2.2: MicroNeighborhoodObject

**Do these when you want quick wins!**

### 🟡 Medium Tasks (3-5 hours each)
- Task 2.1: MicroObject
- Task 5.1: K-Means
- Task 6.1: Simple segmentation
- Task 7.1: Database
- Task 8.1-8.2: Processors

**Core value, reasonable effort**

### 🔴 Complex Tasks (5-6 hours each)
- Task 6.3: 3D segmentation
- Task 5.2: Hierarchical clustering
- Task 9.1: E2E integration

**High value, requires more effort**

---

## 🚀 Commands You'll Use

```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=MicroObjectTest

# Run with coverage report
mvn clean test jacoco:report

# View coverage report
open target/site/jacoco/index.html

# Run tests in watch mode (with Maven wrapper)
mvn test -Dsurefire.rerunFailingTestsCount=1
```

---

## 📝 Test Template

```java
package vtea.objects;

import org.junit.jupiter.api.*;
import org.mockito.Mock;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

class MicroObjectTest {
    
    private MicroObject microObject;
    
    @BeforeEach
    void setUp() {
        // Initialize test data
        microObject = new MicroObject();
    }
    
    @Test
    void testCentroidCalculation() {
        // Given
        ArrayList<int[]> pixels = createTestPixels();
        
        // When
        MicroObject obj = new MicroObject(pixels, 0, mockStack, 1);
        
        // Then
        assertThat(obj.getCentroidX()).isCloseTo(5.0f, within(0.1f));
        assertThat(obj.getCentroidY()).isCloseTo(5.0f, within(0.1f));
    }
    
    @Test
    void testEmptyObjectThrowsException() {
        // Given
        ArrayList<int[]> emptyPixels = new ArrayList<>();
        
        // Then
        assertThatThrownBy(() -> 
            new MicroObject(emptyPixels, 0, mockStack, 1)
        ).isInstanceOf(IllegalArgumentException.class)
         .hasMessageContaining("empty");
    }
}
```

---

## 🎁 Benefits You'll Get

**After Week 1:**
- Can refactor MicroObject safely
- Exceptions fully validated
- CI/CD running tests automatically

**After Week 2:**
- Clustering algorithm correctness verified
- Database operations reliable
- Measurement accuracy confirmed

**After Week 3:**
- Processor error handling solid
- Can modify segmentation with confidence
- Integration tests catch breaking changes

**After Week 4:**
- 50% coverage across codebase
- Major algorithms battle-tested
- Regression protection in place
- Ready for production deployments

---

## 💡 Pro Tips

1. **Write Tests for Bugs:** When you find a bug, write a failing test first
2. **Keep Tests Fast:** Mock slow operations (I/O, database, UI)
3. **One Assert Per Test:** Makes failures clear
4. **Descriptive Names:** `testCentroidCalculationWithSinglePixel()`
5. **Use Test Data Builders:** Make object creation easy
6. **Don't Test Libraries:** Trust SMILE, test your usage of it
7. **Parallel Tests:** Use `@Execution(CONCURRENT)` for speed

---

## 📦 Dependencies to Add

```xml
<!-- Testing -->
<dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter</artifactId>
    <version>5.10.1</version>
    <scope>test</scope>
</dependency>
<dependency>
    <groupId>org.mockito</groupId>
    <artifactId>mockito-core</artifactId>
    <version>5.8.0</version>
    <scope>test</scope>
</dependency>
<dependency>
    <groupId>org.assertj</groupId>
    <artifactId>assertj-core</artifactId>
    <version>3.25.1</version>
    <scope>test</scope>
</dependency>

<!-- Coverage -->
<plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <version>0.8.11</version>
    <executions>
        <execution>
            <goals>
                <goal>prepare-agent</goal>
            </goals>
        </execution>
        <execution>
            <id>report</id>
            <phase>test</phase>
            <goals>
                <goal>report</goal>
            </goals>
        </execution>
    </executions>
</plugin>
```

---

## ✅ Ready to Start?

**Your first command:**
```bash
# 1. Add dependencies above to pom.xml
# 2. Create test directories
mkdir -p src/test/java
mkdir -p src/test/resources

# 3. Run build
mvn clean test

# 4. See: "Tests run: 0, Failures: 0, Errors: 0, Skipped: 0"
```

Want me to implement **Task 1.1** (Set up test framework) right now?
