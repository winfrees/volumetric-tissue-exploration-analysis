/*
 * Copyright (C) 2025 University of Nebraska
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
package vtea.deeplearning.training;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Metrics and loss functions for training deep learning models.
 *
 * @author VTEA Deep Learning Team
 */
public class Metrics {

    /**
     * Compute cross-entropy loss
     *
     * @param predictions Model output logits [batch, num_classes]
     * @param targets True labels [batch]
     * @return Loss value
     */
    public static double crossEntropyLoss(Tensor predictions, Tensor targets) {
        Tensor loss = cross_entropy_loss(predictions, targets);
        double lossValue = loss.item_double();
        loss.close();
        return lossValue;
    }

    /**
     * Compute weighted cross-entropy loss for imbalanced datasets
     *
     * @param predictions Model output logits [batch, num_classes]
     * @param targets True labels [batch]
     * @param classWeights Weights for each class
     * @return Loss value
     */
    public static double weightedCrossEntropyLoss(Tensor predictions, Tensor targets, double[] classWeights) {
        // Convert class weights to tensor
        float[] weightsFloat = new float[classWeights.length];
        for (int i = 0; i < classWeights.length; i++) {
            weightsFloat[i] = (float) classWeights[i];
        }

        Tensor weightTensor = from_blob(weightsFloat, new long[]{classWeights.length});
        weightTensor = weightTensor.to(predictions.device());

        // Compute weighted cross-entropy
        CrossEntropyLossOptions options = new CrossEntropyLossOptions();
        options.weight(weightTensor);

        CrossEntropyLossImpl lossFunc = new CrossEntropyLossImpl(options);
        Tensor loss = lossFunc.forward(predictions, targets);

        double lossValue = loss.item_double();

        loss.close();
        weightTensor.close();

        return lossValue;
    }

    /**
     * Compute accuracy
     *
     * @param predictions Model output logits [batch, num_classes]
     * @param targets True labels [batch]
     * @return Accuracy (0 to 1)
     */
    public static double accuracy(Tensor predictions, Tensor targets) {
        // Get predicted classes
        Tensor predictedClasses = argmax(predictions, 1);

        // Compare with targets
        Tensor correct = predictedClasses.eq(targets);

        // Calculate accuracy
        double accuracy = correct.sum().item_double() / correct.size(0);

        predictedClasses.close();
        correct.close();

        return accuracy;
    }

    /**
     * Compute balanced accuracy (average of per-class accuracies)
     * This metric is important for imbalanced datasets and is used in NephNet3D
     *
     * @param predictions Model output logits [batch, num_classes]
     * @param targets True labels [batch]
     * @param numClasses Number of classes
     * @return Balanced accuracy (0 to 1)
     */
    public static double balancedAccuracy(Tensor predictions, Tensor targets, int numClasses) {
        // Get predicted classes
        Tensor predictedClasses = argmax(predictions, 1);

        // Convert to arrays for easier processing
        long[] predictedArray = tensorToLongArray(predictedClasses);
        long[] targetArray = tensorToLongArray(targets);

        predictedClasses.close();

        // Calculate per-class accuracy
        int[] classCorrect = new int[numClasses];
        int[] classTotal = new int[numClasses];

        for (int i = 0; i < targetArray.length; i++) {
            int trueClass = (int) targetArray[i];
            int predClass = (int) predictedArray[i];

            classTotal[trueClass]++;
            if (trueClass == predClass) {
                classCorrect[trueClass]++;
            }
        }

        // Calculate balanced accuracy
        double sumAccuracy = 0;
        int validClasses = 0;

        for (int i = 0; i < numClasses; i++) {
            if (classTotal[i] > 0) {
                sumAccuracy += (double) classCorrect[i] / classTotal[i];
                validClasses++;
            }
        }

        return validClasses > 0 ? sumAccuracy / validClasses : 0;
    }

    /**
     * Compute confusion matrix
     *
     * @param predictions Model output logits [batch, num_classes]
     * @param targets True labels [batch]
     * @param numClasses Number of classes
     * @return Confusion matrix [numClasses][numClasses]
     */
    public static int[][] confusionMatrix(Tensor predictions, Tensor targets, int numClasses) {
        Tensor predictedClasses = argmax(predictions, 1);

        long[] predictedArray = tensorToLongArray(predictedClasses);
        long[] targetArray = tensorToLongArray(targets);

        predictedClasses.close();

        int[][] matrix = new int[numClasses][numClasses];

        for (int i = 0; i < targetArray.length; i++) {
            int trueClass = (int) targetArray[i];
            int predClass = (int) predictedArray[i];
            matrix[trueClass][predClass]++;
        }

        return matrix;
    }

    /**
     * Compute precision, recall, and F1 score for each class
     *
     * @param confusionMatrix Confusion matrix
     * @return Map containing precision, recall, and F1 for each class
     */
    public static Map<String, double[]> classificationMetrics(int[][] confusionMatrix) {
        int numClasses = confusionMatrix.length;

        double[] precision = new double[numClasses];
        double[] recall = new double[numClasses];
        double[] f1Score = new double[numClasses];

        for (int i = 0; i < numClasses; i++) {
            // True positives
            int tp = confusionMatrix[i][i];

            // False positives
            int fp = 0;
            for (int j = 0; j < numClasses; j++) {
                if (j != i) {
                    fp += confusionMatrix[j][i];
                }
            }

            // False negatives
            int fn = 0;
            for (int j = 0; j < numClasses; j++) {
                if (j != i) {
                    fn += confusionMatrix[i][j];
                }
            }

            // Calculate metrics
            precision[i] = (tp + fp > 0) ? (double) tp / (tp + fp) : 0;
            recall[i] = (tp + fn > 0) ? (double) tp / (tp + fn) : 0;
            f1Score[i] = (precision[i] + recall[i] > 0) ?
                        2 * precision[i] * recall[i] / (precision[i] + recall[i]) : 0;
        }

        Map<String, double[]> metrics = new HashMap<>();
        metrics.put("precision", precision);
        metrics.put("recall", recall);
        metrics.put("f1", f1Score);

        return metrics;
    }

    /**
     * Compute macro-averaged precision, recall, and F1
     *
     * @param metrics Classification metrics from classificationMetrics()
     * @return Map containing macro-averaged values
     */
    public static Map<String, Double> macroAverageMetrics(Map<String, double[]> metrics) {
        double[] precision = metrics.get("precision");
        double[] recall = metrics.get("recall");
        double[] f1 = metrics.get("f1");

        double avgPrecision = average(precision);
        double avgRecall = average(recall);
        double avgF1 = average(f1);

        Map<String, Double> macroMetrics = new HashMap<>();
        macroMetrics.put("macro_precision", avgPrecision);
        macroMetrics.put("macro_recall", avgRecall);
        macroMetrics.put("macro_f1", avgF1);

        return macroMetrics;
    }

    /**
     * Helper: Convert tensor to long array
     */
    private static long[] tensorToLongArray(Tensor tensor) {
        long size = tensor.size(0);
        long[] array = new long[(int) size];

        LongPointer ptr = tensor.data_ptr_long();
        for (int i = 0; i < size; i++) {
            array[i] = ptr.get(i);
        }

        return array;
    }

    /**
     * Helper: Calculate average of array
     */
    private static double average(double[] array) {
        if (array.length == 0) return 0;

        double sum = 0;
        for (double val : array) {
            sum += val;
        }
        return sum / array.length;
    }

    /**
     * Print confusion matrix
     */
    public static String formatConfusionMatrix(int[][] matrix, String[] classNames) {
        int numClasses = matrix.length;
        StringBuilder sb = new StringBuilder();

        // Header
        sb.append("\nConfusion Matrix:\n");
        sb.append("True \\ Pred  ");
        for (int i = 0; i < numClasses; i++) {
            String name = (classNames != null && i < classNames.length) ?
                         classNames[i] : "Class_" + i;
            sb.append(String.format("%-12s", name.substring(0, Math.min(name.length(), 10))));
        }
        sb.append("\n");

        // Rows
        for (int i = 0; i < numClasses; i++) {
            String name = (classNames != null && i < classNames.length) ?
                         classNames[i] : "Class_" + i;
            sb.append(String.format("%-12s  ", name.substring(0, Math.min(name.length(), 10))));

            for (int j = 0; j < numClasses; j++) {
                sb.append(String.format("%-12d", matrix[i][j]));
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    /**
     * Print classification report
     */
    public static String formatClassificationReport(int[][] confusionMatrix, String[] classNames) {
        Map<String, double[]> metrics = classificationMetrics(confusionMatrix);
        Map<String, Double> macroMetrics = macroAverageMetrics(metrics);

        double[] precision = metrics.get("precision");
        double[] recall = metrics.get("recall");
        double[] f1 = metrics.get("f1");

        StringBuilder sb = new StringBuilder();
        sb.append("\nClassification Report:\n");
        sb.append(String.format("%-15s %-10s %-10s %-10s %-10s\n",
                               "Class", "Precision", "Recall", "F1-Score", "Support"));
        sb.append("-".repeat(60)).append("\n");

        int numClasses = confusionMatrix.length;
        for (int i = 0; i < numClasses; i++) {
            String name = (classNames != null && i < classNames.length) ?
                         classNames[i] : "Class_" + i;

            int support = 0;
            for (int j = 0; j < numClasses; j++) {
                support += confusionMatrix[i][j];
            }

            sb.append(String.format("%-15s %-10.3f %-10.3f %-10.3f %-10d\n",
                                   name, precision[i], recall[i], f1[i], support));
        }

        sb.append("-".repeat(60)).append("\n");
        sb.append(String.format("%-15s %-10.3f %-10.3f %-10.3f\n",
                               "Macro Avg",
                               macroMetrics.get("macro_precision"),
                               macroMetrics.get("macro_recall"),
                               macroMetrics.get("macro_f1")));

        return sb.toString();
    }

    /**
     * Training history tracker
     */
    public static class History {
        private java.util.List<Double> trainLoss = new java.util.ArrayList<>();
        private java.util.List<Double> trainAccuracy = new java.util.ArrayList<>();
        private java.util.List<Double> validationLoss = new java.util.ArrayList<>();
        private java.util.List<Double> validationAccuracy = new java.util.ArrayList<>();
        private java.util.List<Double> validationBalancedAccuracy = new java.util.ArrayList<>();
        private java.util.List<Double> learningRates = new java.util.ArrayList<>();

        public void recordEpoch(double trainLoss, double trainAcc,
                               double valLoss, double valAcc, double valBalAcc,
                               double lr) {
            this.trainLoss.add(trainLoss);
            this.trainAccuracy.add(trainAcc);
            this.validationLoss.add(valLoss);
            this.validationAccuracy.add(valAcc);
            this.validationBalancedAccuracy.add(valBalAcc);
            this.learningRates.add(lr);
        }

        public double getBestValidationAccuracy() {
            return validationAccuracy.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        }

        public double getBestValidationBalancedAccuracy() {
            return validationBalancedAccuracy.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        }

        public int getBestEpoch() {
            double bestAcc = getBestValidationBalancedAccuracy();
            for (int i = 0; i < validationBalancedAccuracy.size(); i++) {
                if (Math.abs(validationBalancedAccuracy.get(i) - bestAcc) < 1e-6) {
                    return i;
                }
            }
            return -1;
        }

        // Getters
        public java.util.List<Double> getTrainLoss() { return trainLoss; }
        public java.util.List<Double> getTrainAccuracy() { return trainAccuracy; }
        public java.util.List<Double> getValidationLoss() { return validationLoss; }
        public java.util.List<Double> getValidationAccuracy() { return validationAccuracy; }
        public java.util.List<Double> getValidationBalancedAccuracy() { return validationBalancedAccuracy; }
        public java.util.List<Double> getLearningRates() { return learningRates; }
    }
}
