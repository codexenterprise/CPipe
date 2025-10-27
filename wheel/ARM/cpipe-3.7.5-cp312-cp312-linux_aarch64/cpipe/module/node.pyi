from cpipe.config.config import *
from _typeshed import Incomplete
from cpipe.module.cbuffer import CBuffer as CBuffer
from cpipe.module.cdata import CData as CData, CImage as CImage
from cpipe.module.cevent import CEventQueue as CEventQueue
from cpipe.module.clogger import CLogger as CLogger
from cpipe.module.cmask import CMask as CMask
from cpipe.module.cqueue import BaseQueue as BaseQueue, CQueue as CQueue
from cpipe.module.security import Security as Security
from typing import Any, Callable

class Node:
    TYPE_Streamer: str
    TYPE_Other: str
    TYPE_Algorithm: str
    TYPE_Insight: str
    TYPE_Logic: str
    TYPE_Report: str
    TYPE_Nodes: str
    NODE_STATE_ALIVE: int
    NODE_STATE_DEAD: int
    CPIPE_EXIT_WAIT_TIME: float
    MAX_EVENT_QUEUE_SIZE: int
    EVENT_UPDATE_MASK: str
    __allNodes__: Incomplete
    __PAIRS__: Incomplete
    __NEED_RESTART__: Incomplete
    __NEED_EXIT__: Incomplete
    logger: Incomplete
    __CPIPE_PID__: Incomplete
    __LAUNCH_CONFIG_PATH__: Incomplete
    __exiting__: Incomplete
    nodeName: Incomplete
    bandingStreamerList: Incomplete
    inputs: list[BaseQueue] | None
    nodeType: Incomplete
    queue_size: Incomplete
    queue: CQueue | None
    event_queue: Incomplete
    event_callback: Incomplete
    group_node: Incomplete
    parent_node_name: Incomplete
    processor: Incomplete
    daemon: bool
    event_thread: Incomplete
    pid: Incomplete
    exit_code: Incomplete
    __ready: Incomplete
    state: Incomplete
    death_num: int
    special_mask: Incomplete
    dump_frame_history: Incomplete
    startup_functions: Incomplete
    def __init__(self, nodeName, queue_size: int = 0, *args, **kwargs) -> None:
        """
        CPipe node base class.
        Args:
            nodeName: nodeName: (str) The name of the node.
            queue_size: (int) The size of the queue.
        """
    def on_startup(order: int = 0):
        """
        decorator: a function that takes another function as input and returns a new function with additional functionality.
        Args:
            order: the order of the function, the smaller the number, the earlier the execution, the default is 0
        """
    def event(event_name):
        """
        decorator: a function that takes another function as input and returns a new function with additional functionality.
        Args:
            event_name: (str) The name of the event.
        """
    def event_register(self, event: str, callback: Callable[[Any], None]):
        """
        Register the event callback function.
        Args:
            event: (str) The event name.
            callback: (Callable[[Any], None]) The callback function.
        """
    def update_mask(self, data) -> None:
        """
        Update the mask of the streamer.
        Args:
            data: (Any) The data. The data need to be serializable(mutilprocessing.Queue).
        """
    def _event_thread(self) -> None:
        """
        The event thread.
        """
    def event_send(self, event: str, data: Any | None = None):
        """
        Send the event to the event thread.
        Args:
            event: (str) The event name.
            data: (Any) The data. The data need to be serializable(mutilprocessing.Queue).
        Returns: (Any) The return value of the event callback function.
        """
    def preset_mask(self, mask_name, mask_type, required: bool = False) -> None:
        '''
        Set the special mask for the node.
        Args:
            mask_name: (str) The name of the mask. Note: The mask name must be unique. if required is True, the mask name will be changed to __{mask_name}__
            mask_type: (str) The type of the mask. e.g. "polygons" or "lines"
            required: (bool) Whether the mask is required.
        '''
    @classmethod
    def save_all_nodes(cls, file_path) -> None: ...
    @classmethod
    def load_all_nodes(cls, file_path) -> None: ...
    def lastly(self, signum, frame) -> None:
        """
        The last function to be executed before the node is terminated.
        Args:
            signum: (int) The signal number.
            frame: (frame) The frame.

        Returns:

        """
    def close(self, exit_flag: bool = False) -> None:
        """
        Close the node, the node will be clear shared memory, if used share memory.
        Returns:

        """
    def ready(self) -> None:
        """
        Set the node to ready state.
        Returns None
        """
    def unready(self) -> None:
        """
        Set the node to unready state.
        Returns: None

        """
    def is_ready(self):
        """
        Check if the node is ready.
        Returns: (bool) True or False.

        """
    def get_pid(self):
        """
        Get the PID of the node.
        Returns: (int) The PID of the node.

        """
    def get_exitcode(self):
        """
        Get the exit code of the node.
        Returns: (int) The exit code of the node.

        """
    def get_state(self):
        """
        Get the state of the node. 0: alive(Node.NODE_STATE_ALIVE), 1: dead(Node.NODE_STATE_DEAD)
        Returns:

        """
    def get_frames(self, wait_time: int = 0, block: bool = True, need_all_frames: bool = False):
        """
        Get all frames from the input queue.
        Args:
            wait_time: (float)In blocking mode（block=True）, this parameter takes effect, at least one frame is obtained, otherwise it will wait for wait_time seconds before returning.
            block: (bool) Whether to block.
            need_all_frames: (bool) Whether to get all frames.

        Returns: (CData) The data of the frames.

        """
    def get_streamer_frame(self, node_name=None, cimage: CImage = None):
        """
        Get the frame from the streamer.
        Args:
            node_name: (str) The name of the streamer.
            cimage: (CImage) The image object.

        Returns: (np.ndarray) The frame, (float) The timestamp when this frame was generated.

        """
    def get_cbuffer_with_streamer_name(self, streamer_name):
        """
        Get the cbuffer with the streamer name.
        Args:
            streamer_name: (str) The name of the streamer.
        """
    def __create_node_buffer(self, queue_size, buffer_size=None) -> None:
        """
        Create the node buffer.
        Args:
            queue_size: (int) The size of the queue.
            buffer_size: (tuple) The size of the buffer.

        Returns: None

        """
    def set_input_queue(self, queues: list[BaseQueue]):
        """
        Set the input queue for the current node, these queues will subscribe to the output queue of the current node.
        Args:
            queues: (List[BaseQueue]) The input queue.

        Returns:

        """
    def feed_to(self, nodes: list['Node']):
        """
        Feed to the list of input nodes, for example: [node1, node2, node3], all input nodes will subscribe to the output queue of the current node.
        Args:
            nodes: (List['Node']) The input nodes. For example: [node1, node2, node3]

        Returns: self

        """
    def __str__(self) -> str:
        """
        Get the string of the node.
        Returns: (str) The string of the node.

        """
    def __add__(self, others):
        """
        Overload the addition operator, add the current node to other nodes, that is, the other nodes are the subsequent nodes of the self node.
        Args:
            others: (Node) The input node.

        Returns: self

        """
    def start(self) -> None:
        """
        Start the node, the node will start a new process to run.
        Returns: None

        """
    def _start(self) -> None:
        """
        The entry function for the current node (process) to run, all the logic processing of the current node is completed here.
        Returns: None

        """
    @classmethod
    def get_captain_names(cls):
        """
        Get all the captain names. Captain: The node that has the Streamer node as the input.
        Returns: (List[str]) The list of captain names.

        """
    def get_my_captain_names(self):
        """
        Get the captain names of the current node. Captain: The node that has the Streamer node as the input.
        Returns: (List[str]) The list of captain names.

        """
    @classmethod
    def terminate(cls, exit_flag: bool = False, clear_nodes: bool = False) -> None:
        """
        Terminate all nodes of CPipe.
        Args:
            clear_nodes: (bool) Whether to clear all nodes.
            exit_flag: (bool) Whether to exit CPipe.

        Returns:

        """
    @classmethod
    def restart(cls, delay: int = 1) -> None:
        """
        Restart CPipe.
        Args:
            delay: (int) The delay time before restarting.

        Returns:

        """
    @classmethod
    def exit(cls, info=None, delay: int = 1) -> None:
        """
        Exit CPipe.
        Args:
            info: (str) The exit information.
            delay: (int) The delay time before exiting

        Returns:

        """
    @staticmethod
    def import_node(package_str, parameters):
        """
        Import the node class from the package string and create the node object.
        Args:
            package_str: (str) The package string.
            parameters: (dict) The parameters of the node.

        Returns: (Node) The node object.

        """
    @classmethod
    def create_link(cls) -> None:
        '''
        Create node connection relationships and initialize nodes based on the configuration file (launch.yaml).
        launch.yaml path: configured through Node.launch(launch_config_path="./launch.yaml")
        Returns: None

        '''
    @classmethod
    def get_module_name(cls, streams):
        """
        Determine the class module name of the streamer that needs to be imported through the video stream address or file name address.
        Args:
            streams: (List[str]) The stream address or file name address.

        Returns:

        """
    @classmethod
    def add_link(cls, node_name, streams, captain_names):
        """
        Add a link to the configuration file (launch.yaml) and initialize the node.
        Args:
            node_name: (str) The name of the node.
            streams: (List[str]) The stream address or file name address.
            captain_names: (List[str]) The captain names.

        Returns: (bool, str) Whether the operation is successful, the reason for the failure.

        """
    @classmethod
    def modify_link(cls, node_name, streams, captain_names):
        """
        Modify the link in the configuration file (launch.yaml) and reinitialize the node.
        Args:
            node_name: (str) The name of the node.
            streams: (List[str]) The stream address or file name address.
            captain_names: (List[str]) The captain names.

        Returns: (bool, str) Whether the operation is successful, the reason for the failure.

        """
    @classmethod
    def del_link(cls, node_name):
        """
        Delete the link in the configuration file (launch.yaml) and reinitialize the node.
        Args:
            node_name: (str) The name of the node.

        Returns: (bool, str) Whether the operation is successful, the reason for the failure.

        """
    @classmethod
    def handler(cls, signum, frame) -> None: ...
    @classmethod
    def register_signal(cls) -> None: ...
    def mcp_tool(name):
        """
        decorator: a function that takes another function as input and returns a new function with additional functionality.
        Args:
            name: (str) The name of the mcp tool
        """
    def get_info(self) -> str:
        """
        Get the information of the node.
        Returns: (str) The information of the node.
        """
    @classmethod
    def get_nodes_info(cls, node_name=None) -> str:
        """
        Get all nodes information or specific node information of CPipe.
        Args:
            node_name: (str) The name of the node. default: None, return all nodes information.

        Returns: (str) The information of all nodes or a specific node.
        """
    @classmethod
    def get_nodes_mask(cls, mask_name=None, node_name=None):
        '''
        Get all nodes mask or specific node mask. The mask is a "polygon"(ROI/area) or "line".
        Args:
            mask_name: (str) The name of the detection area of the node, or the name of the mask, or the name of the node valid area.
            node_name: (str) The name of the node. default: None, return all nodes mask.
        Returns: (str) The mask of the node. mask coords[[x1, y1], [x2, y2], ...], x and y are normalized to [0, 1].
        '''
    @classmethod
    def set_node_mask(cls, mask_name, node_name=None, streamer_name=None, mask_type=None, mask_coords=None, delete: bool = False):
        '''
        Set the mask("polygon"(ROI/area) or "line") of the node. streamer_name, node_name and mask_name must be provided at least one.
        Args:
            mask_name: (str) The name of the detection area of the node, or the name of the mask, or the name of the node valid area.
            node_name: (str) The name of the node.
            streamer_name: (str) The name of the streamer.
            mask_type: (str) The type of the mask. "polygon" or "line".
            mask_coords: (List[List[float]]) The coordinates of the mask. coords[[x1, y1], [x2, y2], ...], x and y are normalized to [0, 1].
            delete: (bool) Whether to delete the mask.
        Returns: (str) The result of the operation.
        '''
    @classmethod
    def system_prompt(cls):
        """
        Get the system prompt of the CPipe MCP Server.
        Returns: (str) The system prompt of the CPipe MCP Server.
        """
    @classmethod
    def mcp_register(cls, port: int = 9988, host: str = '0.0.0.0') -> None: ...
    @classmethod
    def launch(cls, check_node: bool = True, check_interval: int = 60, auto_restart: bool = True, launch_config_path=None, auto_handle_signal: bool = True, agent: bool = False, mcp_port: int = 9988, mcp_host: str = '0.0.0.0') -> None:
        """
        Launch all nodes that have been initialized (all nodes that inherit the Node class).
        Args:
            auto_handle_signal: (bool) Whether to automatically handle the signal.
            check_node: (bool) Turn on the node detection function, restart it if a node dies. If you receive a restart signal, restart the entire cpipe.
            check_interval: (int) Node detection interval time(seconds). Only effective when check_node is True.
            auto_restart: (bool) Whether to automatically restart the node when the node dies. Only effective when check_node is True.
            launch_config_path: The path of the configuration file (launch.yaml).
                                launch.yaml : The configuration file of the node connection relationship and the initialization parameters of the node.
            agent: (bool) Whether to enable the agent mode.

        Returns: None

        """
