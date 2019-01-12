package test;

import org.junit.Test;
import utils.Log;
import weka.WekaUtils;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static weka.WekaUtils.*;

/**
 * Description:
 *
 * @author Hao Fu(haofu@ucdavis.edu)
 * @since 2019/1/9
 */
public class RealData {
    private final static Log log = Log.getLogger(RealData.class);

    private class ModelParameters {
        int attNum;
        boolean smote;
        String smoteClass;
        int smotePercent;
        boolean attriSel;

        ModelParameters(int attNum, boolean smote, String smoteClass, int smotePercent, boolean attriSel) {
            this.attNum = attNum;
            this.smote = smote;
            this.smoteClass = smoteClass;
            this.smotePercent = smotePercent;
            this.attriSel = attriSel;
        }
    }

    @Test
    public void cameraData() throws Exception {
        ModelParameters modelParameters = new ModelParameters(500, true, "2", 130, false);
        eval("Camera", true, "D:\\workspace\\COSPOS_MINING", false, modelParameters);
    }

    @Test
    public void reaData() throws Exception {
        ModelParameters modelParameters = new ModelParameters(100, true, "1", 230, true);
        eval("REA", true, "D:\\workspace\\COSPOS_MINING", false, modelParameters);
    }

    @Test
    public void recData() throws Exception {
        ModelParameters modelParameters = new ModelParameters(500, false, "1", 230, false);
        eval("REC", true, "D:\\workspace\\COSPOS_MINING", false, modelParameters);
    }

    @Test
    public void senData() throws Exception {
        ModelParameters modelParameters = new ModelParameters(500, true, "1", 830, false);
        eval("SEN", true, "D:\\workspace\\COSPOS_MINING", false, modelParameters);
    }

    @Test
    public void locData() throws Exception {
        ModelParameters modelParameters = new ModelParameters(500, false, "1", 230, false);
        RealData.eval("Location", true, "D:\\workspace\\COSPOS_MINING", false, modelParameters);
    }

    @Test
    public void locUserData() throws Exception {
        ModelParameters modelParameters = new ModelParameters(500, false, "1", 230, false);
        eval("Location", true, "D:\\workspace\\COSPOS_MINING", true, modelParameters);
    }

    private static void eval(String perm, boolean prediction, String pyWorkLoc, boolean user, ModelParameters modelParameters) throws Exception {
        // boolean user = false;
        List<String> labels = new ArrayList<>();
        String feature_type;

        WekaUtils wekaUtils = new WekaUtils();

        // String perm = "SEND_SMS";//Location";//RECORD_AUDIO";/";//Camera"; //READ_PHONE_STATE";//SEND_SMS";//BLUETOOTH";//Location";//";//NFC"; //";//"; Camera"; //"; //"; //Location"; //; ";//Location"; //"; //"; //SEND_SMS";
        boolean smote = modelParameters.smote;
        boolean attriSel = modelParameters.attriSel;
        int attNum = modelParameters.attNum;
        String smoteClass = modelParameters.smoteClass;
        int smotePercent = modelParameters.smotePercent;

        List<List<WekaUtils.LabelledDoc>> docsResutls = new ArrayList<>();
        if (!user) {
            feature_type = "full";
            //Instances ata = weka.WekaUtils.loadArff();
            //FilteredClassifier filteredClassifier = weka.WekaUtils.buildClassifier(data);
            //System.out.println(filteredClassifier.getBatchSize());

            docsResutls.add(wekaUtils.getDocs(pyWorkLoc + "\\output\\gnd\\comp\\" + perm + "\\"
                    + feature_type));// D:\\workspace\\COSPOS_MINING\\output\\gnd\\" + PERMISSION); //Location");
            labels.add("T");
            //labels.add("D");
            labels.add("F");
        } else {
            feature_type = "users"; //READ_PHONE_STATE";
            for (int i = 0; i < 10; i++) {
                perm = Integer.toString(i);
                docsResutls.add(wekaUtils.getUserDocs(pyWorkLoc + "\\output\\gnd\\" + feature_type + "\\" + i)); //Location");
            }
            labels = new ArrayList<>();
            labels.add("T");
            labels.add("F");
        }

        Map<Integer, Double> f_measures = new HashMap<>();
        for (int i = 0; i < docsResutls.size(); ++i) {
            List<WekaUtils.LabelledDoc> labelledDocs = docsResutls.get(i);
            Instances instances = docs2Instances(labelledDocs, labels);
            log.debug(instances.numInstances());
            if (instances.numInstances() < 10) {
                continue;
            }
            for (Instance instance : instances) {
                log.debug(instance.classAttribute());
                log.debug(instance);
            }

            StringToWordVector stringToWordVector = getWordFilter(instances, false);

            instances = Filter.useFilter(instances, stringToWordVector);
            AttributeSelection attributeSelection = null;

            if (attriSel) {
                attributeSelection = getAttributeSelector(instances, attNum);
                instances = attributeSelection.reduceDimensionality(instances);
            }

            createArff(instances, feature_type + "_" + perm + ".arff");
        /*PrintWriter out = new PrintWriter(PERMISSION + "_" + perm + ".arff");
        out.print(instances.toString());
        out.close();*/
            weka.core.SerializationHelper.write(feature_type + "_" + perm + ".filter", stringToWordVector);
            if (!user && smote) {
                instances = WekaUtils.overSampling(instances, smoteClass, smotePercent); //250);
                log.info(instances.numInstances());

                createArff(instances, feature_type + "_" + perm + "_smote.arff");
            }

            // Evaluate classifier and print some statistics
            Classifier classifier = buildClassifier(instances, feature_type + "_" + perm, true);

            try {
                f_measures.put(i, crossValidation(instances, classifier, 5));
            } catch (Exception e) {
                e.printStackTrace();
            }

            if (prediction) {
                List<WekaUtils.LabelledDoc> labelledTestDocs = wekaUtils.getDocs("data/test");
                Instances testInstances = docs2Instances(labelledTestDocs, labels);

                testInstances = Filter.useFilter(testInstances, stringToWordVector);
                if (attriSel) {
                    testInstances = attributeSelection.reduceDimensionality(testInstances);
                }
                // Evaluate classifier and print some statistics
                Evaluation eval = new Evaluation(instances);
                eval.evaluateModel(classifier, testInstances);
                log.info(eval.toSummaryString("\nResults\n======\n", false));
                log.info(eval.toClassDetailsString());
                log.info(eval.toMatrixString());

                List<String> unlabelledDocs = new ArrayList<>();
                unlabelledDocs.add("ancisco;4:45 AM PDT;Severe Weather Alert;;Severe Weather Alert;;Severe Weather Alert;;More;Rain;73°;57°;60°;;;New York;7:45 AM EDT;Severe Weather Alert;;Severe Weather Alert;;Severe Weather Alert;;More;Sign in;Edit Locations;;Open;Free;San Francisco;;Open;Free;New York;;Open;Free;Tools;Settings;;Open;Free;Send feedback;;Open;Free;Share this app;;Open;Free;Rate this app;;Open;Free;Terms and Privacy");
                predict(unlabelledDocs, stringToWordVector, classifier, instances.classAttribute());
            }
            // save2Arff(instances, "data_bag");
            // save2Arff(testInstances, "test_bag");
        }
    }


    public static void main(String[] args) throws Exception {
//        optionsOptions options = new Options();
//
//        Option input = new Option("i", "input", true, "input file path");
//        input.setRequired(true);
//        options.addOption(input);
//
//        Option output = new Option("o", "output", true, "output file");
//        output.setRequired(true);
//        options.addOption(output);
//
//        CommandLineParser parser = new DefaultParser();
//        HelpFormatter formatter = new HelpFormatter();
//        CommandLine cmd;
//
//        try {
//            cmd = parser.parse(options, args);
//        } catch (ParseException e) {
//            System.out.println(e.getMessage());
//            formatter.printHelp("utility-name", options);
//            System.exit(1);
//        }
//
//        eval();
//
//        FileInputStream fileInputStream = new FileInputStream(new File("full_Location.model"));
//        Classifier classifier = (Classifier) SerializationHelper.read(fileInputStream);
//        fileInputStream = new FileInputStream(new File("full_Location.filter"));
//        StringToWordVector stringToWordVector = loadStr2WordVec(fileInputStream);
//
//        List<String> unlabelledDocs = new ArrayList<>();
//        unlabelledDocs.add("Severe Weather Alert;;Severe Weather Alert;;More;Rain;73°;57°;60°;;;New York;7:45 AM EDT;Severe Weather Alert;;Severe Weather Alert;;Severe Weather Alert;;More;Sign in;Edit Locations;;Open;Free;San Francisco;;Open;Free;New York;;");
//        predict(unlabelledDocs, stringToWordVector, classifier, null);
        RealData realData = new RealData();
        realData.locData();
    }
}
