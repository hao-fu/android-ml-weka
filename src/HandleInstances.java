import weka.WekaUtils;
import weka.core.Instances;

/**
 * Description:
 *
 * @author Hao Fu(haofu@ucdavis.edu)
 * @since 4/25/2017
 */
public class HandleInstances {
    public static void main(String[] args) throws Exception {
        String mark = "RECORD_AUDIO"; //SEND_SMS";
        String PERMISSION = "full";
        Instances instances = WekaUtils.readArff(PERMISSION + "_" + mark + ".arff", 0);
        System.out.println(instances.numInstances());
        instances = WekaUtils.overSampling(instances, "1", 250);
        System.out.println(instances.numInstances());
        instances = WekaUtils.overSampling(instances, "2", 150);
        System.out.println(instances.numInstances());

        WekaUtils.createArff(instances, PERMISSION + "_" + mark + "_smote.arff");
    }
}
