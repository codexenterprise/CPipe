from _typeshed import Incomplete
from collections.abc import Generator
from cpipe.config.config import SHARE_MEMORY_MAX_HEIGHT_WIDTH_CHANNEL as SHARE_MEMORY_MAX_HEIGHT_WIDTH_CHANNEL
from cpipe.module.cbuffer import CSharedImageBuffer as CSharedImageBuffer
from cpipe.module.cmask import CMask as CMask
from cpipe.module.node import Node as Node
from cpipe.module.security import Security as Security
from cpipe.module.utils import ChineseText as ChineseText
from multiprocessing import sharedctypes as sharedctypes

class Insight(Node):
    SHOW_TYPE_UI: int
    SHOW_TYPE_HTTP: int
    SHOW_TYPE_NODE: int
    instances: Incomplete
    show_type: Incomplete
    show_scale: Incomplete
    _streamers: Incomplete
    streamer_messages: Incomplete
    show_chinese: Incomplete
    streamer_frame_delay: Incomplete
    kwargs: Incomplete
    save_video: Incomplete
    video_writers: Incomplete
    def __init__(self, nodeName, queue_size, show_type=..., show_scale: int = 2, save_video: bool = False, **kwargs) -> None:
        """
        The node used to display the video stream in CPipe, supports UI display, HTTP display, and saving video streams. This is a basic class.
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The size of the queue.
            show_type: (int) The type of the display, 0: UI, 1: HTTP, 2: Save.
            show_scale: (int) The scale of the display.
            save_video: (bool) Whether to save the video stream.
            **kwargs: Other parameters.
        """
    def init_streamer_list(self) -> None:
        """
        Initialize the streamer list and initialize the shared memory. This method is called before the start method.
        Returns:

        """
    def delay_update(self, streamer_name, m_time, current_time):
        """
        Update the delay time of the frame of the streamer.
        Args:
            streamer_name: Streamer Node name.
            m_time: The time of the frame.
            current_time: The current time.

        Returns: The fps and the delay time.

        """
    def lastly(self, signum, frame) -> None: ...
    @staticmethod
    def get_show_args(kwargs=None, get_default: bool = False): ...
    def event_show_args(self, show_args=None):
        """
        The event function of the show_args parameter of CPipeInsight.
        Args:
            show_args: The show arguments.
        Returns: None
        """
    def set_show_arguments(self, show_args: dict):
        """
        Set the `show_args` parameter of CPipeInsight. This parameter is used to draw the boxes(or line/polygon/circle) and text on the screen that display the execution results of each node.
        show_args need a dict. Note: Color is BGR format.
        Args:
            show_polygon_box: (bool) Whether to show the polygon box.
            show_box: (bool) Whether to show the box.
            show_box_name: (bool) Whether to show the box name.
            show_text: (bool) Whether to show the text.
            show_polygon: (bool) Whether to show the polygon.
            show_mask: (bool) Whether to show the mask.
            show_key_points: (bool) Whether to show the key points.
            show_person: (bool) Whether to show the person.
            show_classification: (bool) Whether to show the classification.
            show_track: (bool) Whether to show the track.
            chinese_font_size: (int) The size of the Chinese font.
            key_points_name_font_scale: (float) The scale of the key points name font.
            key_points_color: (tuple) The color of the key points.
            key_points_name_color: (tuple) The color of the key points name.
            mask_color: (tuple) The color of the mask.
            track_id_font_scale: (float) The scale of the track id font.
            track_id_color: (tuple) The color of the track id.
            box_name_color: (tuple) The color of the box name.
            box_color: (tuple) The color of the box.
            embedding_box_name_color: (tuple) The color of the embedding box name.
            classification_name_color: (tuple) The color of the classification name.
        Returns: None
        """
    @staticmethod
    def draw_bboxes(show_chinese, show, img_show, one_box, show_args) -> None: ...
    def save_video_thread(self, out, queue_one) -> None: ...
    show_args: Incomplete
    def _start(self) -> None:
        """
        The entry function for the current node (process) to run in one-stage mode, all the logic processing of the current node is completed here.

        Returns: None

        """

class CPipeInsight(Insight):
    log: Incomplete
    app: Incomplete
    current_file_path: Incomplete
    current_directory: Incomplete
    is_running: bool
    ip: str
    port: int
    http_insight: bool
    show_type: Incomplete
    ssl: Incomplete
    cert_path: Incomplete
    key_path: Incomplete
    fps_push: Incomplete
    interval: Incomplete
    dead_img: Incomplete
    def __init__(self, nodeName: str = 'CPipeInsight', queue_size: int = 3, show_scale: int = 1, show_fps: int = 25, ip=None, port=None, http_insight: bool = False, ui_insight: bool = False, save_video: bool = False, save_streamer_node_name=None, save_file_names=None, save_path: str = './save_stream', save_duration_seconds: int = 86400, save_fps: int = 25, save_wh=(1920, 1080), auto_exit: bool = False, ssl: bool = False, cert_path=None, key_path=None, **kwargs) -> None:
        '''
        Used to display the video stream in CPipe, and support HTTP display.
        Args:
            nodeName: The name of the node.
            queue_size: The size of the queue.
            show_scale: The scale of the display. eg: 2 is 1/2 scale.
            show_fps: The show fps.
            ip: The ip address.
            port: The port.
            http_insight: Whether to enable HTTP display.
            ui_insight: Whether to enable UI display, if True, http_insight must be False.
            save_video: Whether to save the video stream.
            save_streamer_node_name: The streamer node name to save, if None, save all streamer nodes. （Due to performance issues, try to save only one video stream.）
            save_file_names: The save file name. If None, the file name is the streamer node name + random number. format: {nodeName<Streamer>: file_name, ...}
            save_path: The save path.
            save_duration_seconds: The save duration in seconds.
            save_fps: The save fps.
            save_wh: The save width and height.
            auto_exit: Auto exit when all streamer nodes are dead.
            show_polygon_box: self.kwargs.get("show_polygon_box", False)
            show_box: self.kwargs.get("show_box", True)
            show_box_name: self.kwargs.get("show_box_name", True)
            show_text: self.kwargs.get("show_text", True)
            show_polygon: self.kwargs.get("show_polygon", True)
            show_mask: self.kwargs.get("show_mask", True)
            show_key_points: self.kwargs.get("show_key_points", True)
            show_person: self.kwargs.get("show_person", True)
            show_classification: self.kwargs.get("show_classification", True)
            show_track: self.kwargs.get("show_track", True)

            chinese_font_size = self.kwargs.get("font size", 20)
            key_points_name_font_scale = self.kwargs.get("key_points_name_font_scale", 1)
            key_points_color = self.kwargs.get("key_points_color", (0, 255, 128))
            key_points_name_color = self.kwargs.get("key_points_name_color", (128, 255, 128))
            mask_color = self.kwargs.get("mask_color", (0, 255, 0))
            track_id_font_scale = self.kwargs.get("track_id_font_scale", 1)
            track_id_color = self.kwargs.get("track_id_color", (0, 0, 255))
            box_name_color = self.kwargs.get("box_name_color", (255, 128, 0))
            classification_name_color = self.kwargs.get("classification_name_color", (0, 255, 0))
            embedding_box_name_color = self.kwargs.get("embedding_box_name_color", (0, 255, 200))
            ssl: Whether to use https.
            cert_path: The cert path with https.
            key_path: The key path with https.

        '''
    processor: Incomplete
    def start(self) -> None:
        """
        Start the node.
        Returns:

        """
    def node(*args):
        """
        The node page.
        Args:
            *args:

        Returns: The node page.

        """
    def get_data(*args):
        """
        Get all Node information.
        Args:
            *args:

        Returns: The Node information.

        """
    def load_node_info(*args):
        """
        Load all Node link information.
        Args:
            *args:

        Returns:

        """
    def index(*args):
        """
        The index page.
        Args:
            *args:

        Returns: The index page.

        """
    def restart(*args):
        """
        Restart the cpipe.
        Args:
            *args:

        Returns: The response data.

        """
    def connect(*args):
        """
        Connect the cpipe, and get the cpipe streamer information.
        Args:
            *args:

        Returns: The response data.

        """
    def del_streamer(*args):
        """
        Delete the streamer.
        Args:
            *args:

        Returns: The response data.

        """
    def set_streamer(*args):
        """
        Add the streamer.
        Args:
            *args:

        Returns: The response data.

        """
    def modify_streamer(*args):
        """
        Modify the streamer.
        Args:
            *args:

        Returns: The response data.

        """
    def get_device_info(*args):
        """
        Get the device information.
        Args:
            *args:

        Returns: The response data.

        """
    def get_image(*args):
        """
        Get the streamer image.
        Args:
            *args:

        Returns: The response data.

        """
    def set_cmask(*args):
        """
        Set the cmask. The cmask is used to mask the streamer image. The cmask can be a polygon or a line.
        Args:
            *args:

        Returns: The response data.

        """
    def del_cmask(*args):
        """
        Delete the cmask.
        Args:
            *args:

        Returns: The response data.

        """
    def static_files(*args, filename):
        """
        Get the static files.
        Args:
            *args:
            filename:

        Returns: The static files.

        """
    def show(nodeName, **kwargs):
        """
        Show the streamer image.
        Args:
            nodeName: The node name.
            **kwargs:

        Returns: The streamer image.

        """
    def insight(nodeName, **kwargs):
        """
        Show the insight page.
        Args:
            nodeName: The node name.
            **kwargs:

        Returns: The insight page.

        """
    def debug_message(nodeName, *args):
        """
        Get the debug message of the node.
        Args:
            *args:
            nodeName: The node name.

        Returns: The debug message.

        """
    def generate_frames(self, stream_name) -> Generator[Incomplete]:
        """
        Generate the streamer image from the streamer.
        Args:
            stream_name: The streamer name.

        Returns: The streamer image.

        """
    def get_current_show_image(self, stream_name):
        """
        Get the current streamer image with all algorithm results.
        Args:
            stream_name: The streamer name.

        Returns: The streamer image or None if the streamer is dead.

        """
