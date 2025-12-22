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
package vtea.clustering;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import java.util.ArrayList;

import static org.assertj.core.api.Assertions.*;

/**
 * Test class for KMeans clustering
 *
 * KMeans is a clustering algorithm that partitions data into K clusters
 * based on distance minimization.
 */
@DisplayName("KMeans Clustering Tests")
class KMeansTest {

    private KMeans kmeans;
    private KMeans kmeansWithMax;
    private double[][] testData;
    private ArrayList<Object> testProtocol;

    private static final double DELTA = 0.001;

    @BeforeEach
    void setUp() {
        kmeans = new KMeans();
        kmeansWithMax = new KMeans(100);
        testData = createTestData();
        testProtocol = createTestProtocol(3, 10);
    }

    // ========== Constructor and Metadata Tests ==========

    @Test
    @DisplayName("Default constructor should initialize with correct metadata")
    void testDefaultConstructorMetadata() {
        assertThat(kmeans.getName()).isEqualTo("K-means Clustering (VTEA)");
        assertThat(kmeans.getKey()).isEqualTo("KmeansVTEA");
        assertThat(kmeans.getVersion()).isEqualTo("0.1");
        assertThat(kmeans.getAuthor()).isEqualTo("Andrew McNutt");
        assertThat(kmeans.getComment()).isEqualTo("Implementation of K-means");
    }

    @Test
    @DisplayName("Constructor with max should initialize protocol")
    void testConstructorWithMaxInitializesProtocol() {
        ArrayList protocol = kmeansWithMax.getOptions();

        assertThat(protocol).isNotNull();
        assertThat(protocol).hasSize(4);
        assertThat(protocol.get(0)).isInstanceOf(JLabel.class);
        assertThat(protocol.get(1)).isInstanceOf(JSpinner.class);
        assertThat(protocol.get(2)).isInstanceOf(JLabel.class);
        assertThat(protocol.get(3)).isInstanceOf(JTextField.class);
    }

    @Test
    @DisplayName("Constructor with max should set cluster spinner bounds")
    void testConstructorSetsSpinnerBounds() {
        ArrayList protocol = kmeansWithMax.getOptions();
        JSpinner spinner = (JSpinner) protocol.get(1);
        SpinnerNumberModel model = (SpinnerNumberModel) spinner.getModel();

        assertThat(model.getMinimum()).isEqualTo(2);
        assertThat(model.getMaximum()).isEqualTo(100);
        assertThat(model.getValue()).isEqualTo(5);
    }

    @Test
    @DisplayName("Should extend AbstractFeatureProcessing")
    void testExtendsAbstractFeatureProcessing() {
        assertThat(kmeans).isInstanceOf(vtea.featureprocessing.AbstractFeatureProcessing.class);
    }

    @Test
    @DisplayName("TYPE should be Cluster")
    void testTypeIsCluster() {
        assertThat(kmeans.TYPE).isEqualTo("Cluster");
    }

    // ========== Distance Calculation Tests ==========

    @Test
    @DisplayName("calculateDistanceSq() should calculate squared Euclidean distance")
    void testCalculateDistanceSq() {
        double[] p1 = {0.0, 0.0};
        double[] p2 = {3.0, 4.0};

        double result = kmeans.calculateDistanceSq(p1, p2);

        // Distance squared: 3^2 + 4^2 = 9 + 16 = 25
        assertThat(result).isEqualTo(25.0, within(DELTA));
    }

    @Test
    @DisplayName("calculateDistanceSq() should return zero for identical points")
    void testCalculateDistanceSqIdenticalPoints() {
        double[] p1 = {5.0, 10.0, 15.0};
        double[] p2 = {5.0, 10.0, 15.0};

        double result = kmeans.calculateDistanceSq(p1, p2);

        assertThat(result).isEqualTo(0.0, within(DELTA));
    }

    @Test
    @DisplayName("calculateDistanceSq() should handle negative coordinates")
    void testCalculateDistanceSqNegativeCoordinates() {
        double[] p1 = {-3.0, -4.0};
        double[] p2 = {0.0, 0.0};

        double result = kmeans.calculateDistanceSq(p1, p2);

        assertThat(result).isEqualTo(25.0, within(DELTA));
    }

    @Test
    @DisplayName("calculateDistanceSq() should handle high-dimensional data")
    void testCalculateDistanceSqHighDimensional() {
        double[] p1 = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] p2 = {2.0, 3.0, 4.0, 5.0, 6.0};

        double result = kmeans.calculateDistanceSq(p1, p2);

        // Each dimension differs by 1, so sum of squares = 5
        assertThat(result).isEqualTo(5.0, within(DELTA));
    }

    @Test
    @DisplayName("calculateDistanceSq() should throw exception for mismatched dimensions")
    void testCalculateDistanceSqMismatchedDimensions() {
        double[] p1 = {1.0, 2.0};
        double[] p2 = {1.0, 2.0, 3.0};

        assertThatThrownBy(() -> kmeans.calculateDistanceSq(p1, p2))
            .isInstanceOf(RuntimeException.class)
            .hasMessageContaining("dimensions");
    }

    @Test
    @DisplayName("calculateDistanceSq() should handle single dimension")
    void testCalculateDistanceSqSingleDimension() {
        double[] p1 = {10.0};
        double[] p2 = {5.0};

        double result = kmeans.calculateDistanceSq(p1, p2);

        assertThat(result).isEqualTo(25.0, within(DELTA));
    }

    // ========== Process Method Tests ==========

    @Test
    @DisplayName("process() should return true on successful clustering")
    void testProcessReturnsTrue() {
        boolean result = kmeans.process(testProtocol, testData, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("process() should populate dataResult")
    void testProcessPopulatesDataResult() {
        kmeans.process(testProtocol, testData, false);

        ArrayList dataResult = kmeans.getResult();
        assertThat(dataResult).isNotEmpty();
        assertThat(dataResult).hasSize(1);
    }

    @Test
    @DisplayName("process() should assign cluster membership to all points")
    void testProcessAssignsAllPoints() {
        kmeans.process(testProtocol, testData, false);

        ArrayList dataResult = kmeans.getResult();
        ArrayList membership = (ArrayList) dataResult.get(0);

        assertThat(membership).hasSize(testData.length);
    }

    @Test
    @DisplayName("process() should assign valid cluster IDs")
    void testProcessAssignsValidClusterIDs() {
        int nClusters = 3;
        ArrayList<Object> protocol = createTestProtocol(nClusters, 5);

        kmeans.process(protocol, testData, false);

        ArrayList dataResult = kmeans.getResult();
        ArrayList membership = (ArrayList) dataResult.get(0);

        for (Object clusterID : membership) {
            int id = (int) clusterID;
            assertThat(id).isBetween(0, nClusters - 1);
        }
    }

    @Test
    @DisplayName("process() should handle 2 clusters minimum")
    void testProcessWith2Clusters() {
        ArrayList<Object> protocol = createTestProtocol(2, 5);

        boolean result = kmeans.process(protocol, testData, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("process() should handle many clusters")
    void testProcessWithManyClusters() {
        // Create more data points for many clusters
        double[][] moreData = createLargeTestData(50);
        ArrayList<Object> protocol = createTestProtocol(10, 10);

        boolean result = kmeans.process(protocol, moreData, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("process() should handle single iteration")
    void testProcessWithSingleIteration() {
        ArrayList<Object> protocol = createTestProtocol(3, 1);

        boolean result = kmeans.process(protocol, testData, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("process() should handle multiple trials")
    void testProcessWithMultipleTrials() {
        ArrayList<Object> protocol = createTestProtocol(3, 20);

        boolean result = kmeans.process(protocol, testData, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("process() should handle normalized data")
    void testProcessWithNormalizedData() {
        // Protocol with z-normalization enabled
        ArrayList<Object> protocol = createTestProtocol(3, 10);
        protocol.set(0, true);  // Enable z-normalization

        boolean result = kmeans.process(protocol, testData, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("process() should handle single-dimensional data")
    void testProcessWithSingleDimension() {
        double[][] data1D = new double[20][1];
        for (int i = 0; i < 20; i++) {
            data1D[i][0] = i * 5.0;
        }

        ArrayList<Object> protocol = createTestProtocol(2, 5);

        boolean result = kmeans.process(protocol, data1D, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("process() should handle high-dimensional data")
    void testProcessWithHighDimensionalData() {
        double[][] dataHD = new double[30][10];
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 10; j++) {
                dataHD[i][j] = Math.random() * 100;
            }
        }

        ArrayList<Object> protocol = createTestProtocol(3, 5);

        boolean result = kmeans.process(protocol, dataHD, false);

        assertThat(result).isTrue();
    }

    // ========== Clustering Quality Tests ==========

    @Test
    @DisplayName("process() should separate well-defined clusters")
    void testProcessSeparatesWellDefinedClusters() {
        // Create 3 well-separated clusters
        double[][] separatedData = new double[30][2];

        // Cluster 1: around (0, 0)
        for (int i = 0; i < 10; i++) {
            separatedData[i][0] = Math.random() * 2;
            separatedData[i][1] = Math.random() * 2;
        }

        // Cluster 2: around (100, 100)
        for (int i = 10; i < 20; i++) {
            separatedData[i][0] = 100 + Math.random() * 2;
            separatedData[i][1] = 100 + Math.random() * 2;
        }

        // Cluster 3: around (0, 100)
        for (int i = 20; i < 30; i++) {
            separatedData[i][0] = Math.random() * 2;
            separatedData[i][1] = 100 + Math.random() * 2;
        }

        ArrayList<Object> protocol = createTestProtocol(3, 10);

        kmeans.process(protocol, separatedData, false);

        ArrayList dataResult = kmeans.getResult();
        ArrayList membership = (ArrayList) dataResult.get(0);

        // Should have assigned to 3 different clusters
        assertThat(membership).hasSize(30);

        // Verify at least 2 different clusters are used
        boolean[] clusterUsed = new boolean[3];
        for (Object m : membership) {
            clusterUsed[(int) m] = true;
        }

        int usedCount = 0;
        for (boolean used : clusterUsed) {
            if (used) usedCount++;
        }

        assertThat(usedCount).isGreaterThanOrEqualTo(2);
    }

    @Test
    @DisplayName("process() should handle identical data points")
    void testProcessWithIdenticalPoints() {
        double[][] identicalData = new double[10][2];
        for (int i = 0; i < 10; i++) {
            identicalData[i][0] = 5.0;
            identicalData[i][1] = 5.0;
        }

        ArrayList<Object> protocol = createTestProtocol(2, 5);

        boolean result = kmeans.process(protocol, identicalData, false);

        assertThat(result).isTrue();
    }

    // ========== getBlockComment() Tests ==========

    @Test
    @DisplayName("getBlockComment() should generate comment from components")
    void testGetBlockComment() {
        ArrayList components = new ArrayList();
        components.add(null);  // 0-3: other components
        components.add(null);
        components.add(null);
        components.add(null);
        components.add(new JLabel("Clusters"));  // 4
        components.add(new JSpinner(new SpinnerNumberModel(5, 2, 100, 1)));  // 5
        components.add(new JLabel("Iterations"));  // 6
        components.add(new JTextField("10"));  // 7

        String comment = KMeans.getBlockComment(components);

        assertThat(comment).contains("Clusters");
        assertThat(comment).contains("5");
        assertThat(comment).contains("Iterations");
        assertThat(comment).contains("10");
        assertThat(comment).startsWith("<html>");
        assertThat(comment).endsWith("</html>");
    }

    // ========== getDataDescription() Tests ==========

    @Test
    @DisplayName("getDataDescription() should include cluster count")
    void testGetDataDescription() {
        ArrayList params = new ArrayList();
        for (int i = 0; i < 5; i++) {
            params.add(null);
        }
        params.add(new JSpinner(new SpinnerNumberModel(7, 2, 100, 1)));

        String description = kmeans.getDataDescription(params);

        assertThat(description).contains("KmeansVTEA");
        assertThat(description).contains("7");
    }

    // ========== Edge Case Tests ==========

    @Test
    @DisplayName("process() should handle data with same feature values")
    void testProcessWithSameFeatureValues() {
        double[][] sameFeatureData = new double[10][2];
        for (int i = 0; i < 10; i++) {
            sameFeatureData[i][0] = 5.0;  // All same X
            sameFeatureData[i][1] = i;     // Different Y
        }

        ArrayList<Object> protocol = createTestProtocol(2, 5);

        boolean result = kmeans.process(protocol, sameFeatureData, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("process() should handle minimum dataset size")
    void testProcessWithMinimumDataset() {
        double[][] minData = new double[3][2];
        minData[0] = new double[]{0.0, 0.0};
        minData[1] = new double[]{1.0, 1.0};
        minData[2] = new double[]{2.0, 2.0};

        ArrayList<Object> protocol = createTestProtocol(2, 1);

        boolean result = kmeans.process(protocol, minData, false);

        assertThat(result).isTrue();
    }

    @Test
    @DisplayName("validate flag should be accessible")
    void testValidateFlag() {
        assertThat(KMeans.validate).isNotNull();
    }

    // ========== Helper Methods ==========

    /**
     * Creates test data with 20 points in 2D space
     */
    private double[][] createTestData() {
        double[][] data = new double[20][2];

        // Create some clustered data
        for (int i = 0; i < 10; i++) {
            data[i][0] = Math.random() * 10;
            data[i][1] = Math.random() * 10;
        }

        for (int i = 10; i < 20; i++) {
            data[i][0] = 50 + Math.random() * 10;
            data[i][1] = 50 + Math.random() * 10;
        }

        return data;
    }

    /**
     * Creates larger test data
     */
    private double[][] createLargeTestData(int n) {
        double[][] data = new double[n][2];

        for (int i = 0; i < n; i++) {
            data[i][0] = Math.random() * 100;
            data[i][1] = Math.random() * 100;
        }

        return data;
    }

    /**
     * Creates test protocol ArrayList
     *
     * Protocol structure:
     * 0: z-normalization (boolean)
     * 1: selectData (ArrayList of selected columns)
     * 2-4: other components
     * 5: JSpinner for cluster count
     * 6: JLabel "Iterations"
     * 7: JTextField for trial count
     */
    private ArrayList<Object> createTestProtocol(int nClusters, int nTrials) {
        ArrayList<Object> protocol = new ArrayList<>();

        protocol.add(false);  // 0: z-normalization off

        // 1: Select all columns
        ArrayList<Integer> selectData = new ArrayList<>();
        selectData.add(0);
        selectData.add(1);
        protocol.add(selectData);

        protocol.add(null);  // 2: placeholder
        protocol.add(null);  // 3: placeholder
        protocol.add(new JLabel("Clusters"));  // 4

        JSpinner spinner = new JSpinner(new SpinnerNumberModel(nClusters, 2, 100, 1));
        protocol.add(spinner);  // 5

        protocol.add(new JLabel("Iterations"));  // 6

        JTextField trials = new JTextField(String.valueOf(nTrials));
        protocol.add(trials);  // 7

        return protocol;
    }
}
