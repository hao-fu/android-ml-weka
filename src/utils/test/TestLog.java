package utils.test;

import org.junit.Before;
import org.junit.Test;
import utils.Log;

/**
 * Description:
 *
 * @author Hao Fu(haofu@ucdavis.edu)
 * @since 2019/1/10
 */
public class TestLog {
    private static Log log;

    @Before
    public void init() {
        log = Log.getLogger("TestLog", Log.DEBUG);
    }

    @Test
    public void testInfo() {
        log.info("TestInfo");
    }

    @Test
    public void testDebug() {
        log.debug("TestDebug");
    }

    @Test
    public void testWarn() {
        log.warning("TestWarning");
    }

    @Test
    public void testError() {
        log.error("TestError");
    }
}
