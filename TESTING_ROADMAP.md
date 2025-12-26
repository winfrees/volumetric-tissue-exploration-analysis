# VTEA Unit Testing Roadmap

**Goal:** Achieve meaningful test coverage incrementally, starting with highest-value components.

**Target Coverage:** 50% overall (starting from 0%)

---

## Phase 1: Infrastructure & Foundation (1-2 days)

### Task 1.1: Set Up Test Framework ⏱️ 2-3 hours
**Priority:** P1 - Blocking all other tests

**Subtasks:**
- [ ] Add JUnit 5 (Jupiter) to pom.xml
- [ ] Add Mockito 5.x for mocking
- [ ] Add AssertJ for fluent assertions
- [ ] Configure Maven Surefire plugin
- [ ] Create test directory structure (`src/test/java`)
- [ ] Create test resources directory (`src/test/resources`)

**Dependencies to add:**
```xml
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
```

**Deliverable:** `mvn test` runs successfully (even with 0 tests)

---

### Task 1.2: Create Test Base Classes ⏱️ 1-2 hours
**Priority:** P1 - Needed for consistency

**Subtasks:**
- [ ] Create `BaseTest.java` with common test utilities
- [ ] Create `BaseProcessorTest.java` for processor tests
- [ ] Create test helper utilities (mock data generators)
- [ ] Create sample test data files (small TIFF images)
- [ ] Write documentation on test patterns to follow

**Deliverable:** Reusable test infrastructure ready

---

### Task 1.3: Write First Smoke Test ⏱️ 30 min
**Priority:** P1 - Validates setup

**Subtasks:**
- [ ] Create `VTEASmokeTest.java`
- [ ] Test that main class loads
- [ ] Test that SciJava context initializes
- [ ] Verify basic constants are set correctly

**Deliverable:** First passing test, CI/CD can be configured

---

## Phase 2: Core Data Models (2-3 days)

### Task 2.1: Test MicroObject ⏱️ 4-6 hours
**Priority:** P1 - Most important data structure
**File:** `vteaobjects/MicroObject.java`

**Test Coverage:**
- [ ] Constructor tests (various pixel configurations)
- [ ] Centroid calculation accuracy
- [ ] Feature storage/retrieval
- [ ] Morphological operations
- [ ] Serialization/deserialization
- [ ] Edge cases (empty objects, single pixel, large objects)

**Target Coverage:** 70%+ for this class

---

### Task 2.2: Test MicroNeighborhoodObject ⏱️ 2-3 hours
**Priority:** P2
**File:** `vteaobjects/MicroNeighborhoodObject.java`

**Test Coverage:**
- [ ] Neighbor relationship management
- [ ] Distance calculations
- [ ] Spatial indexing
- [ ] Edge cases (no neighbors, many neighbors)

**Target Coverage:** 60%+

---

### Task 2.3: Test Dataset Classes ⏱️ 2-3 hours
**Priority:** P2
**Files:** `vtea/dataset/Dataset.java`, `ImageRepository.java`

**Test Coverage:**
- [ ] Dataset creation and initialization
- [ ] Image addition/removal
- [ ] Metadata management
- [ ] Feature storage

**Target Coverage:** 50%+

---

## Phase 3: Exception Handling (1 day)

### Task 3.1: Test Custom Exceptions ⏱️ 2-3 hours
**Priority:** P1 - We just created these!
**Files:** All classes in `vtea/exceptions/`

**Test Coverage:**
- [ ] VTEAException construction (all variants)
- [ ] ImageProcessingException with proper messages
- [ ] SegmentationException exception chaining
- [ ] DatabaseException serialization
- [ ] ClusteringException stack trace preservation

**Target Coverage:** 95%+ (exceptions are easy to test)

---

## Phase 4: Utility & Helper Classes (2-3 days)

### Task 4.1: Test Image Processing Utilities ⏱️ 3-4 hours
**Priority:** P2
**Files:** `vtea/imageprocessing/builtin/*.java`

**Test Coverage:**
- [ ] Gaussian filtering with various sigma values
- [ ] Median filtering correctness
- [ ] Background subtraction accuracy
- [ ] Edge detection validation
- [ ] Input validation (null images, wrong dimensions)

**Target Coverage:** 60%+

---

### Task 4.2: Test Morphological Operations ⏱️ 2-3 hours
**Priority:** P2
**Files:** `vtea/objects/morphology/*.java`

**Test Coverage:**
- [ ] Grow operations (6C, 26C, Cross)
- [ ] Ring creation
- [ ] Erosion/dilation correctness
- [ ] Edge cases (object at image boundary)

**Target Coverage:** 60%+

---

### Task 4.3: Test Measurement Classes ⏱️ 3-4 hours
**Priority:** P2
**Files:** `vtea/objects/measurements/*.java`

**Test Coverage:**
- [ ] Mean calculation accuracy
- [ ] Standard deviation correctness
- [ ] Min/max detection
- [ ] Threshold mean calculation
- [ ] Edge cases (empty regions, single pixel)

**Target Coverage:** 70%+

---

## Phase 5: Clustering Algorithms (3-4 days)

### Task 5.1: Test K-Means Clustering ⏱️ 4-5 hours
**Priority:** P1 - Core algorithm
**File:** `vtea/clustering/KMeans.java`, `KMeansClust.java`

**Test Coverage:**
- [ ] Clustering with known data (2D Gaussian blobs)
- [ ] K=2,3,5 cluster verification
- [ ] Convergence criteria
- [ ] Empty cluster handling
- [ ] Deterministic results with same seed
- [ ] Edge cases (K=1, K > n_points)

**Target Coverage:** 60%+

---

### Task 5.2: Test Hierarchical Clustering ⏱️ 4-5 hours
**Priority:** P2
**Files:** `vtea/clustering/hierarchical/*.java`

**Test Coverage:**
- [ ] Ward linkage correctness
- [ ] Single linkage correctness
- [ ] Complete linkage correctness
- [ ] Dendrogram construction
- [ ] Distance matrix calculations
- [ ] Known dataset validation

**Target Coverage:** 60%+

---

### Task 5.3: Test Gaussian Mixture Model ⏱️ 3-4 hours
**Priority:** P2
**File:** `vtea/clustering/GaussianMix.java`

**Test Coverage:**
- [ ] EM algorithm convergence
- [ ] Likelihood calculations
- [ ] Component initialization
- [ ] Known distribution fitting
- [ ] Edge cases (singular covariance matrices)

**Target Coverage:** 50%+

---

### Task 5.4: Test X-means & G-means ⏱️ 2-3 hours
**Priority:** P3
**Files:** `vtea/clustering/Xmeans.java`, `GMeansClust.java`

**Test Coverage:**
- [ ] Automatic K selection
- [ ] BIC/AIC calculations
- [ ] Model selection criteria
- [ ] Known optimal K verification

**Target Coverage:** 50%+

---

## Phase 6: Segmentation Algorithms (4-5 days)

### Task 6.1: Test Simple Segmentation ⏱️ 3-4 hours
**Priority:** P1
**File:** `vtea/objects/Segmentation/SingleThreshold.java`

**Test Coverage:**
- [ ] Threshold calculation (Otsu, manual)
- [ ] Object detection in synthetic images
- [ ] Binary mask generation
- [ ] Edge cases (empty image, all foreground)

**Target Coverage:** 60%+

---

### Task 6.2: Test 2D Region Segmentation ⏱️ 3-4 hours
**Priority:** P2
**File:** `vtea/objects/Segmentation/Region2DSingleThreshold.java`

**Test Coverage:**
- [ ] Connected component detection
- [ ] Region growing accuracy
- [ ] Small object filtering
- [ ] Known pattern detection (circles, squares)

**Target Coverage:** 50%+

---

### Task 6.3: Test 3D Segmentation ⏱️ 5-6 hours
**Priority:** P1 - Core functionality
**Files:** `LayerCake3DSingleThreshold.java`, `FloodFill3DSingleThreshold.java`

**Test Coverage:**
- [ ] 3D object detection in synthetic volumes
- [ ] Layer-by-layer processing
- [ ] Object merging across slices
- [ ] Memory efficiency validation
- [ ] Known 3D shape detection (spheres, cubes)

**Target Coverage:** 50%+ (complex algorithms)

---

### Task 6.4: Test ROI-Based Segmentation ⏱️ 2-3 hours
**Priority:** P3
**File:** `vtea/objects/Segmentation/ImageJROIBased.java`

**Test Coverage:**
- [ ] ROI import and processing
- [ ] Multiple ROI handling
- [ ] ROI-to-MicroObject conversion
- [ ] Edge cases (overlapping ROIs)

**Target Coverage:** 60%+

---

## Phase 7: Database Layer (2-3 days)

### Task 7.1: Test H2 Database Engine ⏱️ 4-5 hours
**Priority:** P1 - Just updated with logging
**File:** `vtea/jdbc/H2DatabaseEngine.java`

**Test Coverage:**
- [ ] Database creation (in-memory for tests)
- [ ] Table creation
- [ ] Insert operations
- [ ] Query operations
- [ ] Connection pooling
- [ ] Transaction handling
- [ ] Error recovery
- [ ] Database cleanup

**Target Coverage:** 70%+

---

### Task 7.2: Test Data Persistence ⏱️ 2-3 hours
**Priority:** P2

**Test Coverage:**
- [ ] MicroObject storage and retrieval
- [ ] Measurement storage
- [ ] Feature vector persistence
- [ ] Bulk insert performance
- [ ] Data integrity validation

**Target Coverage:** 60%+

---

## Phase 8: Processors (4-5 days)

### Task 8.1: Test SegmentationProcessor ⏱️ 4-5 hours
**Priority:** P1 - Just updated with logging
**File:** `vtea/processor/SegmentationProcessor.java`

**Test Coverage:**
- [ ] Mock segmentation algorithm execution
- [ ] Progress reporting
- [ ] Error handling
- [ ] Cancellation support
- [ ] Result validation

**Target Coverage:** 60%+

---

### Task 8.2: Test FeatureProcessor ⏱️ 3-4 hours
**Priority:** P1
**File:** `vtea/processor/FeatureProcessor.java`

**Test Coverage:**
- [ ] Mock clustering execution
- [ ] Feature extraction
- [ ] Progress tracking
- [ ] Error propagation

**Target Coverage:** 60%+

---

### Task 8.3: Test MeasurementProcessor ⏱️ 3-4 hours
**Priority:** P2
**File:** `vtea/processor/MeasurementProcessor.java`

**Test Coverage:**
- [ ] Mock measurement calculations
- [ ] Result aggregation
- [ ] Progress updates
- [ ] Error handling

**Target Coverage:** 60%+

---

### Task 8.4: Test Other Processors ⏱️ 4-5 hours
**Priority:** P2
**Files:** `ImageProcessingProcessor.java`, `PlotProcessor.java`, etc.

**Test Coverage:**
- [ ] Basic execution paths
- [ ] Error handling
- [ ] Progress reporting
- [ ] Mock dependencies

**Target Coverage:** 50%+

---

## Phase 9: Integration Tests (2-3 days)

### Task 9.1: End-to-End Segmentation Pipeline ⏱️ 4-5 hours
**Priority:** P1

**Test Coverage:**
- [ ] Load synthetic image
- [ ] Run segmentation
- [ ] Extract measurements
- [ ] Store in database
- [ ] Verify results
- [ ] Test with real small sample image

**Target Coverage:** N/A (integration test)

---

### Task 9.2: End-to-End Clustering Pipeline ⏱️ 3-4 hours
**Priority:** P2

**Test Coverage:**
- [ ] Load measurements
- [ ] Run clustering
- [ ] Store classifications
- [ ] Verify cluster assignments

**Target Coverage:** N/A (integration test)

---

### Task 9.3: Database Integration Tests ⏱️ 2-3 hours
**Priority:** P2

**Test Coverage:**
- [ ] Full workflow with database
- [ ] Concurrent access
- [ ] Large dataset handling
- [ ] Performance benchmarks

**Target Coverage:** N/A (integration test)

---

## Phase 10: Continuous Improvement (Ongoing)

### Task 10.1: Increase Coverage in Low Areas ⏱️ Ongoing
**Priority:** P3

**Activities:**
- [ ] Identify uncovered code paths with coverage reports
- [ ] Add tests for edge cases discovered in production
- [ ] Refactor untestable code
- [ ] Document complex test scenarios

---

### Task 10.2: Performance Benchmarks ⏱️ 3-4 hours
**Priority:** P3

**Activities:**
- [ ] Create JMH benchmark suite
- [ ] Benchmark critical algorithms
- [ ] Establish performance baselines
- [ ] Track performance regression

---

## Quick Start Tasks (Do These First!)

### Week 1: Foundation
1. ✅ Task 1.1: Set up test framework (2-3 hrs)
2. ✅ Task 1.2: Create base classes (1-2 hrs)
3. ✅ Task 1.3: Write smoke test (30 min)
4. ✅ Task 3.1: Test exceptions (2-3 hrs) ← Easy wins!
5. ✅ Task 2.1: Test MicroObject (4-6 hrs)

**Total: ~10-14 hours** | **Coverage: ~5-10%**

### Week 2: Core Algorithms
6. ✅ Task 5.1: Test K-Means (4-5 hrs)
7. ✅ Task 6.1: Test simple segmentation (3-4 hrs)
8. ✅ Task 7.1: Test database (4-5 hrs)
9. ✅ Task 4.3: Test measurements (3-4 hrs)

**Total: ~14-18 hours** | **Coverage: ~20-25%**

### Week 3: Processors & Integration
10. ✅ Task 8.1: Test SegmentationProcessor (4-5 hrs)
11. ✅ Task 8.2: Test FeatureProcessor (3-4 hrs)
12. ✅ Task 9.1: E2E segmentation pipeline (4-5 hrs)
13. ✅ Task 4.1: Test image processing (3-4 hrs)

**Total: ~14-18 hours** | **Coverage: ~35-40%**

### Week 4: Polish & Coverage
14. ✅ Task 5.2: Test hierarchical clustering (4-5 hrs)
15. ✅ Task 6.3: Test 3D segmentation (5-6 hrs)
16. ✅ Task 2.2: Test MicroNeighborhoodObject (2-3 hrs)
17. ✅ Additional coverage tasks as needed

**Total: ~11-14 hours** | **Coverage: ~45-50%**

---

## Effort Summary

| Phase | Tasks | Total Time | Coverage Gain |
|-------|-------|------------|---------------|
| 1. Infrastructure | 3 | 4-6 hours | 0% → 1% |
| 2. Core Data | 3 | 8-12 hours | 1% → 8% |
| 3. Exceptions | 1 | 2-3 hours | 8% → 10% |
| 4. Utilities | 3 | 8-11 hours | 10% → 18% |
| 5. Clustering | 4 | 13-17 hours | 18% → 28% |
| 6. Segmentation | 4 | 13-17 hours | 28% → 38% |
| 7. Database | 2 | 6-8 hours | 38% → 42% |
| 8. Processors | 4 | 14-18 hours | 42% → 48% |
| 9. Integration | 3 | 9-12 hours | 48% → 50% |
| **TOTAL** | **27 tasks** | **77-104 hours** | **0% → 50%** |

**Realistic Timeline:** 3-4 weeks (20-25 hours/week)

---

## Success Metrics

### After Week 1:
- ✅ Test framework working
- ✅ 5-10% coverage
- ✅ CI/CD can run tests
- ✅ Core data model tested

### After Week 2:
- ✅ 20-25% coverage
- ✅ Main algorithms tested
- ✅ Database layer tested
- ✅ Critical bugs found and fixed

### After Week 3:
- ✅ 35-40% coverage
- ✅ All processors tested
- ✅ E2E pipeline working
- ✅ Confidence in refactoring

### After Week 4:
- ✅ 45-50% coverage
- ✅ Complex algorithms tested
- ✅ Integration tests passing
- ✅ Performance baselines established

---

## Tips for Success

1. **Start Small:** Infrastructure first, then easy wins (exceptions, utilities)
2. **Test What Matters:** Focus on algorithms and data models, not UI
3. **Mock External Dependencies:** Don't test SMILE library, test your usage of it
4. **Use Real Test Data:** Create small synthetic images for realistic tests
5. **Write Tests First When Fixing Bugs:** Reproduce the bug in a test, then fix
6. **Don't Aim for 100%:** 50% of well-chosen coverage is better than 90% of everything
7. **Run Tests Often:** Use `mvn test` frequently during development
8. **Track Coverage:** Use JaCoCo plugin to see what's covered

---

## JaCoCo Coverage Setup

Add to pom.xml:
```xml
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

Then run:
```bash
mvn clean test
# View report: target/site/jacoco/index.html
```

---

## Ready to Start?

**Recommended First Day:**
- Task 1.1: Set up test framework (2-3 hrs)
- Task 1.2: Create base classes (1-2 hrs)
- Task 1.3: Write smoke test (30 min)

**Total: 4-6 hours** → You'll have a working test infrastructure!

Want me to implement any of these tasks?
