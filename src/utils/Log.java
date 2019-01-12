package utils;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Description:
 *
 * @author Hao Fu(haofu@ucdavis.edu)
 * @since 2019/1/10
 */
public class Log {
    private String TAG;
    private int level;
    private Logger logger;

    public static final int DEBUG = 10;
    public static final int INFO = 20;
    public static final int WARN = 30;
    public static final int ERROR = 40;

    // The nested class to implement singleton
    private static class SingletonHolder {
        private static final Log instance = new Log();
    }

    // Get THE instance
    private static Log getSingleton() {
        return SingletonHolder.instance;
    }

    public static Log getLogger(String TAG, int level) {
        Log log = getLogger(TAG);
        Level l;
        switch (level) {
            case DEBUG:
                l = Level.FINE;
                break;
            case INFO:
                l = Level.INFO;
                break;
            case WARN:
                l = Level.WARNING;
                break;
            default:
                l = Level.SEVERE;
                break;
        }
        log.logger.setLevel(l);
        log.setLevel(level);
        return log;
    }

    public static Log getLogger(String TAG) {
        Log log = getSingleton();
        log.setTAG(TAG);
        log.logger = Logger.getLogger(TAG);
        return log;
    }

    public static Log getLogger(Class cls) {
        return getLogger(cls.getName());
    }

    public static Log getLogger(Class cls, int level) {
        return getLogger(cls.getName(), level);
    }

    public void setTAG(String TAG) {
        this.TAG = TAG;
    }

    public void setTAG(Object obj) {
        if (obj != null) setTAG(obj.toString());
    }

    public void setLevel(int level) {
        this.level = level;
        Level l;
        switch (level) {
            case DEBUG:
                l = Level.FINE;
                break;
            case INFO:
                l = Level.INFO;
                break;
            case WARN:
                l = Level.WARNING;
                break;
            default:
                l = Level.SEVERE;
                break;
        }
        this.logger.setLevel(l);
    }

    public int getLevel() {
        return level;
    }

    public void debug(Object msg) {
        if (level <= DEBUG && msg != null) {
            this.logger.info(msg.toString());
        }
    }

    public void info(Object msg) {
        if (level <= INFO && msg != null) logger.info(msg.toString());
    }

    public void warning(Object msg) {
        if (level <= WARN && msg != null) logger.warning(msg.toString());
    }

    public void error(Object msg) {
        if (level <= ERROR) this.logger.severe(msg.toString());
    }
}
