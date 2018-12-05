package com.example.amine.learn2sign;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class ClassifierService {

    private SVMHelper svmHelper;

    public void initializeClassifier() throws Exception {
        double[][] xTrain = new double[430][];
        double[][] yTrain = {{1}, {0}, {0}, {0}, {0}, {1}, {1}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {1}, {1}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {0}, {0}, {0}, {1}, {1}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {0}, {1}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {1}, {0}, {1}, {0}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1}, {0}, {1}, {1}, {1}, {0}, {0}, {1}, {0}, {1}, {0}, {0}, {1}, {0}, {0}, {0}, {1}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {0}, {1}, {1}, {1}, {1}, {1}, {0}, {0}, {0}, {1}, {0}, {0}, {1}, {1}, {1}, {0}, {1}, {0}, {1}, {1}, {1}, {0}, {1}, {1}, {1}, {0}, {0}, {1}, {1}, {1}, {0}, {0}, {1}, {1}, {1}, {0}, {1}, {1}, {1}, {0}, {1}, {0}, {0}, {0}, {0}, {1}, {0}, {0}, {1}, {1}, {1}, {1}, {0}, {1}, {0}, {1}, {0}, {0}, {1}, {0}, {1}, {1}, {1}, {0}, {0}, {1}, {0}, {1}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {0}, {1}, {1}, {0}, {1}, {0}, {1}, {1}, {1}, {0}, {0}, {1}, {1}, {0}, {1}, {0}, {0}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {0}, {0}, {1}, {1}, {1}, {0}, {0}, {1}, {1}, {1}, {0}, {1}, {1}, {0}, {0}, {0}, {1}, {1}, {0}, {1}, {0}, {0}, {0}, {1}, {1}, {0}, {0}, {0}, {1}, {1}, {1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {1}, {0}, {1}, {0}, {0}, {1}, {0}, {0}, {1}, {0}, {1}, {1}, {0}, {0}, {1}, {1}, {1}, {0}, {1}, {1}, {1}, {1}, {1}, {0}, {1}, {1}, {0}, {0}, {0}, {1}, {0}, {0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}, {1}, {1}, {1}, {0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}, {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1}, {1}, {0}, {1}, {1}, {0}, {0}, {1}, {0}, {1}, {1}, {1}, {1}, {0}, {0}, {1}, {0}, {1}, {0}, {0}, {1}, {1}, {0}, {1}, {1}, {0}, {0}, {0}, {0}, {1}, {0}, {1}, {1}, {0}, {1}, {0}, {1}, {1}, {0}, {1}, {1}, {1}, {1}, {1}, {1}, {0}, {0}, {1}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, {1}, {0}, {0}, {0}, {1}, {0}, {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {0}, {1}, {1}, {0}, {0}, {1}, {0}, {1}, {0}, {0}, {1}, {1}, {1}, {1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {1}, {0}, {0}, {1}, {0}, {1}, {0}, {0}, {0}, {0}, {1}, {1}};

        File file = new File("/Users/achayapathy/Desktop/about_father/test.csv");
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st;
        int i = 0;
        while ((st = br.readLine()) != null) {
            String[] vals = st.split(",");
            double[] record = new double[22];
            int j = 0;
            for (String val : vals) {
                record[j] = Double.valueOf(val);
                j++;
            }
            xTrain[i] = record;
            i++;
        }

        svmHelper = new SVMHelper();
        svmHelper.setXtrain(xTrain);
        svmHelper.setYtrain(yTrain);
        System.out.println("Training in progress");
        svmHelper.svmTrain();
        System.out.println("Training done");
    }

    public List<String> getPredictionForFiles(List<String> inputFiles) throws Exception {
        double[][] xTest = new double[inputFiles.size()][];
        double[][] yTest = new double[inputFiles.size()][];

        for (String inputFile : inputFiles) {
            int i = 0;
            if (inputFile.toLowerCase().contains("father")) {
                double[] temp = {1};
                yTest[i] = temp;
            } else {
                double[] temp = {0};
                yTest[i] = temp;
            }

            File file = new File(inputFile);
            BufferedReader br = new BufferedReader(new FileReader(file));
            String st = br.readLine();
            double[] finalRecord = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            while ((st = br.readLine()) != null) {
                String[] vals = st.split(",");
                int j = 0;
                for (String val : vals) {
                    finalRecord[j] += Double.valueOf(val);
                    j++;
                }
                for (int k = 0; k < finalRecord.length; k++) {
                    finalRecord[k] /= j;
                }
                xTest[i] = finalRecord;
                i++;
            }
        }

        svmHelper.setXtest(xTest);
        svmHelper.setYtest(yTest);
        double[] result = svmHelper.svmPredict();
        for (double temp : result) {
            System.out.println(temp);
        }

        return new ArrayList<String>();
    }

}

