from _typeshed import Incomplete
from cpipe.module.clogger import CLogger as CLogger

class CProcessor:
    logger: Incomplete
    name: Incomplete
    input_queue: Incomplete
    task: Incomplete
    current_task_result: Incomplete
    processor: Incomplete
    def __init__(self, name, task) -> None:
        """
        CProcessor class. Used for multi-process task processing.
        Args:
            name: processor name
            task: task function

        """
    def start(self) -> None:
        """
        Start the process.
        Returns: None

        """

class CProcessors:
    logger: Incomplete
    name: Incomplete
    _processor_num: Incomplete
    _processors: dict[int, CProcessor]
    input_queue: Incomplete
    output_queue: Incomplete
    busy_wait_time: Incomplete
    def __init__(self, name, processor_num, task, queue_size: int = 32, busy_wait_time: float = 0.04) -> None:
        '''
        CPipe process pool class. Used for multi-process task processing.
        
        # input_queue: data format: {"task_name": task_name, "args": {"arg_name": arg_value, ...}}
        # output queue format: {"task_name": task_name, "result": result, "run_time": run_time, "error": error}

        Args:
            name: process pool name
            processor_num: number of processors
            task: task function
            queue_size: input and output queue size
            busy_wait_time: busy wait time

        '''
    def run(self) -> None:
        """
        Run the process pool.
        Returns:

        """
