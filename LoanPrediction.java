import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.io.ObjectOutputStream;
import java.io.FileOutputStream;

public class LoanPrediction {
    public static void main(String[] args) throws Exception {
        String csvFile = "loan_data.csv"; // path to CSV
        // Load CSV
        DataSource source = new DataSource(csvFile);
        Instances data = source.getDataSet();

        // Ensure class index is last column
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // Convert string attributes to nominal (if any)
        StringToNominal stn = new StringToNominal();
        stn.setAttributeRange("first-last");
        stn.setInputFormat(data);
        // You may restrict range to categorical columns only if needed.
        Instances nominalData = Filter.useFilter(data, stn);

        // Replace missing values
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(nominalData);
        Instances clean = Filter.useFilter(nominalData, replaceMissing);

        // Shuffle
        clean.randomize(new Random(1));

        // Train/test split (80/20)
        int trainSize = (int) Math.round(clean.numInstances() * 0.8);
        int testSize = clean.numInstances() - trainSize;
        Instances train = new Instances(clean, 0, trainSize);
        Instances test = new Instances(clean, trainSize, testSize);

        // Build RandomForest
        RandomForest rf = new RandomForest();
        rf.setNumTrees(100);
        rf.buildClassifier(train);

        // Evaluate
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(rf, test);
        System.out.println("=== Evaluation ===");
        System.out.println(eval.toSummaryString());
        System.out.println("Confusion Matrix:");
        double[][] cm = eval.confusionMatrix();
        for (double[] row : cm) {
            for (double v : row) System.out.printf("%7.2f", v);
            System.out.println();
        }
        System.out.println("AUC: " + eval.areaUnderROC(1));

        // Save model
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("loan_rf.model"));
        oos.writeObject(rf);
        oos.flush();
        oos.close();

        System.out.println("Model saved to loan_rf.model");
    }
}
