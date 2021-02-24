import logging
import logging.handlers
import os
import sys
import traceback


class LoggingSingleton(object):
    """
    Use singleton to wrap python logger with your custom handler.
    For example, TimedRotatingFileHandler which generates
    time rotating files.

    Attributes
    ----------
    logger : logging.RootLogger
        The actual Python logger which do the logging
    _instance : LoggingSingleton
        shared instance of this class

    """

    _instance = None  # shared instance

    def __new__(cls, *args, **kwargs):
        # make sure the instance object only allocated once
        if cls._instance is None:
            cls._instance = super(LoggingSingleton, cls).__new__(cls)
            return cls._instance
        else:
            return None

    def __init__(
        self,
        logging_file_name,
        output_dir,
        logging_level=logging.INFO,
        logger_name=None,
        when="W1",
        interval=1,
        backupCount=2,
    ):
        """
        Parameters
        ----------
        logging_file_name : str
            File name of the output log.
        output_dir : str
            Directory which is to save the log file
        logging_level : int
            loggin level (ex: logging.INFO)
        logger_name : str
            Name of the logger in Python facility
        when : str
            specify the type of interval
        interval : int
            interval of file rotating
        backupCount : int
            number of retened backup files.
        """

        if (logger_name is None) and (logging_file_name != ""):
            logger_name = logging_file_name
        elif logger_name is None:
            logger_name = "ScreenLogger"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(message)s")
        if logging_file_name == "":
            # get a default logger which prints informarion to screen
            self.logger = logging.getLogger(logger_name)
            ch = logging.StreamHandler()
            ch.setLevel(logging_level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.setLevel(logging_level)
        else:
            self.logger = logging.getLogger(logger_name)
            fh = logging.handlers.TimedRotatingFileHandler(
                os.path.join(output_dir, logging_file_name + ".log"), when="W1", interval=interval, backupCount=backupCount
            )
            fh.setLevel(logging_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.setLevel(logging_level)

    def exception(self, msg, *args, **kwargs):
        """
        Record exception into the logger
        Parameters
        ----------
        msg : str
            custom message to be put into the log file during exception
        args : list
            arguments to pass to Python logger's exception method
        kwargs : dict
            arguments to pass to Python logger's exception method
        """
        self.logger.exception(msg, *args, **kwargs)

    def info(self, msg, show=True, *args, **kwargs):
        """
        Record info message into the logger
        Parameters
        ----------
        msg : str
            custom message to be put into the log file.
        args : list
            arguments to pass to Python logger's info method
        kwargs : dict
            arguments to pass to Python logger's info method
        """
        if show:
            print('[INFO] %s' % msg)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Record warning message into the logger
        Parameters
        ----------
        msg : str
            custom warning message to be put into the log file.
        args : list
            arguments to pass to Python logger's warning method
        kwargs : dict
            arguments to pass to Python logger's warning method
        """
        print('[WARN] %s' % msg)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, e, *args, **kwargs):
        """
        Record error message into the logger
        Parameters
        ----------
        msg : str
            custom error message to be put into the log file.
        args : list
            arguments to pass to Python logger's error method
        kwargs : dict
            arguments to pass to Python logger's error method
        """
        error_class = e.__class__.__name__ #å–å¾—éŒ¯èª¤é¡žåž‹
        detail = e.args[0] #å–å¾—è©³ç´°å…§å®¹
        cl, exc, tb = sys.exc_info() #å–å¾—Call Stack
        errMsg = ''
        for lastCallStack in traceback.extract_tb(tb):
            # lastCallStack = traceback.extract_tb(tb)[-1] #å–å¾—Call Stackçš„æœ€å¾Œä¸€ç­†è³‡æ–™
            fileName = lastCallStack[0] #å–å¾—ç™¼ç”Ÿçš„æª”æ¡ˆåç¨±
            lineNum = lastCallStack[1] #å–å¾—ç™¼ç”Ÿçš„è¡Œè™Ÿ
            funcName = lastCallStack[2] #å–å¾—ç™¼ç”Ÿçš„å‡½æ•¸åç¨±
            errMsg += "File \"{}\", line {}, in {}: [{}] {}\n".format(fileName, lineNum, funcName, error_class, detail)
        # print(errMsg)
        print('[ERROR] %s' % errMsg)
        self.logger.error(errMsg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Record critical message into the logger
        Parameters
        ----------
        msg : str
            custom message to be put into the log file.
        args : list
            arguments to pass to Python logger's critical method
        kwargs : dict
            arguments to pass to Python logger's critical method
        """
        print('[CRITICAL] %s' % msg)
        self.logger.critical(msg, *args, **kwargs)

    @classmethod
    def get_instance(cls):
        """
        Get the logger instance
        Parameters
        ----------
        """
        if cls._instance is not None:
            return cls._instance
        else:
            raise Exception("ProbeHierLogging has not yet been initialized.")