import threading
from _typeshed import Incomplete
from cpipe.config.config import CLOGER_LEVEL as CLOGER_LEVEL

class CLogger(threading.Thread):
    '''
    Log class, inherited from threading.Thread, used for multi-threaded writing logs.
    multiprocessing serialization is not a problem under linux, but it will report an error under windows (multi-threading can be used).

    Attributes:
    log_queue_size: The size of the log queue
    report_type: "" or "websocket"
    report_info: {"websocket_url": ""}
    report_func: function(msg) -> json
    report_level: "error" or "info" or "debug"
    report_queue: multiprocessing.Queue
    queue: multiprocessing.Queue
    clogger: logging.Logger
    initialized: bool

    '''
    log_queue_size: int
    report_type: str
    report_info: Incomplete
    report_level: Incomplete
    report_queue: Incomplete
    report_func: Incomplete
    queue: Incomplete
    clogger: Incomplete
    initialized: bool
    file_name_mark: str
    logger_level = CLOGER_LEVEL
    @classmethod
    def _setup_logger(cls) -> None:
        """
        Set the dedicated logger configuration to avoid being overridden by third-party libraries
        """
    @classmethod
    def set_file_name_mark(cls, file_name_mark) -> None:
        """
        Set the file name mark
        """
    @classmethod
    def set_logger_level(cls, level) -> None:
        """
        Set the logger level
        """
    daemon: bool
    def __init__(self) -> None:
        """
        Initialize the CLogger class.

        Returns: None

        """
    def __getstate__(self):
        """
        Get the state of the logger

        Returns: The state of the logger
        """
    def __setstate__(self, state) -> None:
        """
        Set the state of the logger
        Args:
            state: The state of the logger

        Returns: None

        """
    @classmethod
    def init_report(cls, report_type: str, report_info: dict, report_func, report_level=('error', 'report')):
        '''
        Initialize the report function
        Args:
            report_type: "" or "websocket"
            report_info: {"websocket_url": "..."}
            report_func: function(msg) -> json
            report_level: "error" or "info" or "debug" or "report"

        Returns: None

        '''
    @classmethod
    def debug(cls, message, block: bool = False) -> None:
        """
        Log the debug message
        Args:
            message: The message to be logged
            block: Whether to block the queue

        Returns: None

        """
    @classmethod
    def info(cls, message, block: bool = False) -> None:
        """
        Log the info message
        Args:
            message: The message to be logged
            block: Whether to block the queue

        Returns: None

        """
    @classmethod
    def warning(cls, message, block: bool = False) -> None:
        """
        Log the warning message
        Args:
            message: The message to be logged
            block: Whether to block the queue

        Returns: None

        """
    @classmethod
    def error(cls, message, block: bool = False) -> None:
        """
        Log the error message
        Args:
            message: The message to be logged
            block: Whether to block the queue

        Returns: None

        """
    @classmethod
    def report(cls, message, block: bool = False) -> None:
        """
        Log the report message
        Args:
            message: The message to be logged
            block: Whether to block the queue

        Returns: None

        """
    @classmethod
    def run_report(cls) -> None:
        """
        Run the report function
        Returns: None

        """
    @classmethod
    def run(cls, *args, **kwargs) -> None:
        """
        Run the logger
        Args:
            *args:
            **kwargs:

        Returns: None

        """
