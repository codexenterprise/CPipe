from _typeshed import Incomplete
from cpipe.config.config import BLOCKING_MODE_TIMEOUT as BLOCKING_MODE_TIMEOUT, CPIPE_BLOCKING_MODE as CPIPE_BLOCKING_MODE, SHARE_MEMORY_MODE as SHARE_MEMORY_MODE
from cpipe.module.clogger import CLogger as CLogger
from multiprocessing import sharedctypes as sharedctypes

class CSharedImageBuffer:
    logger: Incomplete
    name: Incomplete
    max_height: Incomplete
    max_width: Incomplete
    channels: Incomplete
    max_size: Incomplete
    locks: Incomplete
    ready_time: Incomplete
    shared_memory: Incomplete
    current_shape: Incomplete
    _lock: Incomplete
    def __init__(self, name=None, max_height: int = 5000, max_width: int = 5000, channels: int = 3, consumer_names=None) -> None:
        """
        Initialize the shared memory buffer.
        
        Args:
            name: (str) The name of the buffer.
            max_height: (int) The maximum height of the buffer.
            max_width: (int) The maximum width of the buffer.
            channels: (int) The number of channels of the buffer.
            consumer_names: (list) The list of consumer names.
        """
    def is_free(self):
        """
        Check if the buffer is free for the consumer.
        """
    def lock(self):
        """
        Lock the buffer.
        """
    def unlock(self) -> None:
        """
        Unlock the buffer.
        """
    def to_ready(self) -> None:
        """
        Set the buffer to ready.
        """
    def free(self, consumer_name) -> None:
        """
        Unlock the buffer for the consumer.
        """
    def reshape(self, shape):
        """
        Set the new shape of the buffer.
        
        Args:
            shape: (list) The shape of the buffer. [height, width, channel]
        
        Returns:
            None
        """
    def data(self, shape=None):
        """
        Get the numpy array view of the current shape.

        Args:
            shape: (list) The shape of the array.

        Returns:
            (np.ndarray) The numpy array view of the current shape.
        """
    @property
    def shape(self): ...
    @property
    def size(self): ...

class CBuffer:
    name: Incomplete
    share_memory_size: Incomplete
    buffer_length: Incomplete
    consumer_names: Incomplete
    buffers: list[CSharedImageBuffer]
    current_index: int
    def __init__(self, name, share_memory_size, buffer_length, consumer_names) -> None:
        """
        CPipe shared memory buffer class, used to store shared memory data.
        Args:
            name: (str) The name of the buffer.
            share_memory_size: (list) The size of the shared memory. [height, width, channel]
            buffer_length: (int) The number of shared content blocks.
            consumer_names: (list) The list of consumers.
        """
    def malloc(self, value=None, shape=None):
        """
        Malloc the buffer.
        Args:
            value: The value to be set.
            shape: The shape of the value. [height, width, channel]

        Returns: (np.ndarray, buffer index) The shared memory buffer and the index of the buffer.

        """
    def data(self, index, shape=None):
        """
        Get the value of the buffer
        Args:
            index: The index of the buffer.
            shape: The shape of the image data.

        Returns: (np.ndarray) The shared memory buffer.

        """
    def free(self, index, consumer_name) -> None:
        """
        Free the buffer.
        """
