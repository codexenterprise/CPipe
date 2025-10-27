from _typeshed import Incomplete
from cpipe.config.config import BETWEEN_NODES_SHARE_MEMORY_MODE as BETWEEN_NODES_SHARE_MEMORY_MODE, CLOGER_LEVEL as CLOGER_LEVEL, CLOGER_LEVEL_DEBUG as CLOGER_LEVEL_DEBUG, CPIPE_BLOCKING_MODE as CPIPE_BLOCKING_MODE
from cpipe.module.cdata import CData as CData, CImage as CImage

class BaseQueue:
    consumer_name: Incomplete
    producer_name: Incomplete
    queue_name: Incomplete
    maxsize: Incomplete
    queue: Incomplete
    dump_frame_history: Incomplete
    dump_num: Incomplete
    all_num: Incomplete
    def __init__(self, producer_name, consumer_name, maxsize: int) -> None:
        """
        Base queue class in CPipe, used for data transfer between nodes and nodes.
        Args:
            producer_name: producer Node name
            consumer_name: consumer Node name
            maxsize: queue size
        """
    def dump(self, current_time, dump_flag: bool = False) -> None:
        """
        CPipe is used to record the data transfer between nodes and nodes, whether there is data congestion causing the data processing to be overwhelmed and discarded, and the number of discards.
        Args:
            current_time: current frame time
            dump_flag: whether the frame is discarded

        Returns: None

        """
    def get(self, block: bool = True, timeout=None) -> CData | None:
        """
        Get data from the queue.
        Args:
            block: block flag, default True
            timeout: timeout time, default None

        Returns: CData or None

        """
    def put(self, cdata, block: bool = True, timeout=None) -> None:
        """
        Put data into the queue.
        Args:
            cdata: cdata to put
            block: block flag, default True
            timeout: timeout time, default None

        Returns:

        """
    def full(self):
        """
        Check if the queue is full.
        Returns: True or False

        """
    def empty(self):
        """
        Check if the queue is empty.
        Returns: True or False
        """
    def remain(self):
        """
        Get the remaining space in the queue.
        Returns:

        """

class CQueue:
    __NODE_BUFFERS__: Incomplete
    name: Incomplete
    maxsize: Incomplete
    consumer_queue_list: Incomplete
    dump_info: Incomplete
    def __init__(self, name, maxsize: int) -> None:
        """
        CQueue is a class for data transfer between nodes and nodes in CPipe.
        Args:
            name: Node name
            maxsize: queue size
        """
    def put(self, cdata: CData, force: bool = True, wait_once_time: float = 0.001):
        """
        Put data into the queue.
        Args:
            cdata: CData
            force: whether to force the frame to be discarded when the queue is full
            wait_once_time: wait time when the queue is full

        Returns: None

        """
    def put_cdata(self, cdata: CData, force: bool = True, wait_once_time: float = 0.001):
        """
        Put CData into the queue.
        Args:
            cdata: CData
            force: whether to force the frame to be discarded when the queue is full
            wait_once_time: wait time when the queue is full

        Returns: None

        """
    def subscribe(self, name):
        """
        Subscribe to a queue for the next node to subscribe to the queue of the previous node.
        Args:
            name: Node name
        Returns: BaseQueue
        """
    def qsize(self):
        """
        Get the size of the queue.
        Returns: str of queue size

        """
