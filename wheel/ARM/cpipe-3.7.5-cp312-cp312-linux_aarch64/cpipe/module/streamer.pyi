from _typeshed import Incomplete
from cpipe.config.config import SHARE_MEMORY_MODE as SHARE_MEMORY_MODE, VIDEO_CUDA_MODE as VIDEO_CUDA_MODE
from cpipe.module.cdata import CData as CData, CImage as CImage
from cpipe.module.cevent import CEvent as CEvent
from cpipe.module.cmask import CMask as CMask
from cpipe.module.cprocessors import CProcessors as CProcessors
from cpipe.module.insight import Insight as Insight
from cpipe.module.node import Node as Node
from cpipe.module.thirdparty.hikvision import Hikvision as Hikvision
from cpipe.module.utils import ChineseText as ChineseText, cudaSetDevice as cudaSetDevice
from multiprocessing import shared_memory as shared_memory

class Streamer(Node):
    stream: Incomplete
    new_stream: Incomplete
    queue_size: Incomplete
    nodeName: Incomplete
    nodeType: Incomplete
    cmask: CMask
    all_ready: Incomplete
    device: Incomplete
    actual_fps: Incomplete
    _process_frame_interval: Incomplete
    def __init__(self, nodeName, stream, queue_size, all_ready, device: str = 'cuda:0') -> None:
        """
        Streamer is the base class for the streamer node.
        Args:
            nodeName: (str) The name of the node.
            stream: (str) The stream address.
            queue_size: (int) The size of the queue.
            all_ready: (bool) Whether to wait for all consumers to be ready.
            device: (str) The device of the model, CPU(cpu) or GPU(cuda:x).
        """
    @property
    def process_frame_interval(self):
        """
        Get the process frame interval.
        Returns: (int) The interval of processing frames.

        """
    @process_frame_interval.setter
    def process_frame_interval(self, value) -> None:
        """
        Set the process frame interval.
        Args:
            value: (int) The interval of processing frames.

        Returns: None

        """
    def video_resolution_changed(self) -> None:
        """
        Check if the video resolution changed.
        Returns: (bool) True or False
        """
    def _connect(self) -> None:
        """
        Connect to the stream. This function should be implemented in the subclass.
        Returns: None

        """
    def _start(self) -> None:
        """
        Start the streamer node. This function should be implemented in the subclass.
        Returns: None

        """
    @staticmethod
    def get_video_type(path):
        """
        Get the video type.
        Args:
        """
    def check_all_ready(self) -> None:
        """
        Check if all consumers are ready.
        Returns: None

        """

class VideoStreamer(Streamer):
    video_fps: Incomplete
    video_width: Incomplete
    video_height: Incomplete
    video_channels: Incomplete
    _process_frame_interval: Incomplete
    short_connection_delay: Incomplete
    ground_image: Incomplete
    rotate: Incomplete
    sleep_time: Incomplete
    block_mode: Incomplete
    once_mode: Incomplete
    delay_start_time: Incomplete
    dump_time: int
    hikvision_platform: Incomplete
    hikvision_cameras_info: Incomplete
    hikvision: Incomplete
    cmask: Incomplete
    def __init__(self, nodeName, stream, queue_size, process_frame_interval: int = 0, hikvision_platform: bool = False, base_url: str = '', appKey: str = '', appSecret: str = '', short_connection_delay: float = 0.0, sleep_time=None, block_mode: bool = False, once_mode: bool = False, delay_start_time: int = 0, all_ready: bool = True, rotate: int = 0, device: str = 'cuda:0') -> None:
        '''
        VideoStreamer is the base class for the video streamer node.
        Args:
            nodeName: (str) The name of the node.
            stream: (str) The stream address.
            queue_size: (int) The size of the queue.
            process_frame_interval: (int) The interval of processing frames.
            hikvision_platform: (bool) Whether to use the Hikvision platform.
            base_url: (str) The base url of the Hikvision platform.
            appKey: (str) The appKey of the Hikvision platform.
            appSecret: (str) The appSecret of the Hikvision platform.
            short_connection_delay: (float) The short connection delay.
            sleep_time: (float) The sleep time(second, e.g. 0.04s) with video interframe sleep. Just for file mode. e.g. stream="file.mp4"
            block_mode: (bool) Whether to block the frame. Just for file mode. e.g. stream="file.mp4"
            once_mode:  (bool) Whether to run once. Just for file mode. e.g. stream="file.mp4"
            delay_start_time: (float) The delay start time. Just for file mode. e.g. stream="file.mp4"
            all_ready: (bool) Whether to wait for all consumers to be ready.
            rotate: (int) The rotate of the video. 0: no rotate, 90: rotate 90 degrees, 180: rotate 180 degrees, 270: rotate 270 degrees.
            device: (str) The device of the model, CPU(cpu) or GPU(cuda:x).

        '''
    new_stream: Incomplete
    def evet_set_stream(self, data):
        """
        Set the stream.
        Args:
            data: (str) The stream address.

        Returns: None

        """
    def reset_stream(self, stream):
        """
        Reset the stream.
        Args:
            stream: (str) The stream address.

        Returns: None

        """
    def need_reset_stream(self):
        """
        Check if need to reset the stream.
        Returns: (bool) True or False

        """
    def get_hk_camera_rtsp(self, camera_name):
        """
        Get the rtsp stream of the Hikvision platform.
        Args:
            camera_name: (str) The name of the camera.

        Returns: (str) The rtsp stream.

        """
    def get_rtsp_info(self):
        """
        Get the rtsp information.
        Returns: (int, int, int) The video fps, width and height.

        """
    def get_one_image(self):
        """
        Get one image from the stream.
        Returns: (np.ndarray) The image.

        """
    def _connect(self, wait_connect: bool = True):
        """
        Connect to the stream.
        Args:
            wait_connect: (bool) Whether to wait for the connection to be successful.
        Returns: (cv2.VideoCapture, bool) The cv2.VideoCapture object and the flag.

        """
    def check_video_info(self):
        """
        Check the video information.
        Returns: (bool) True or False

        """
    stream: Incomplete
    def stream_start_cpu(self) -> None:
        """
        Start the streamer node with CPU mode.
        Returns: None

        """
    def stream_start_cuda(self) -> None:
        """
        Start the streamer node with CUDA mode.
        Returns: None

        """
    def file_start_cpu(self) -> None:
        """
        Start the streamer node with CPU mode.
        Returns: None

        """
    def file_start_cuda(self) -> None:
        """
        Start the streamer node with CUDA mode.
        Returns: None

        """
    def _start(self) -> None:
        """
        Start the streamer node.
        Returns: None

        """

class VideoStreamers(Node):
    nodeType: Incomplete
    streams: Incomplete
    processor_num: Incomplete
    interval_time: Incomplete
    round_interval_time: Incomplete
    dead_thres_time: Incomplete
    process_pool_queue_size: Incomplete
    daemon: bool
    def __init__(self, nodeName, streams, queue_size, processor_num, interval_time: float = 0.0, round_interval_time: float = 0.0, hikvision_platform: bool = False, base_url: str = '', appKey: str = '', appSecret: str = '', dead_thres_time: int = 5, process_pool_queue_size: int = 32, all_ready: bool = True, device: str = 'cuda:0') -> None:
        """
        VideoStreamers is the base class for the video streamers node.
        Args:
            nodeName: (str) The name of the node.
            streams: (list) The stream addresses.
            queue_size: (int) The size of the queue.
            processor_num: (int) The number of processors.
            interval_time: (float) The interval time. Each streamer will be processed every interval_time seconds.
            round_interval_time: (float) The round interval time(seconds).
            hikvision_platform: (bool) Whether to use the Hikvision platform.
            base_url: (str) The base url of the Hikvision platform.
            appKey: (str) The appKey of the Hikvision platform.
            appSecret: (str) The appSecret of the Hikvision platform.
            dead_thres_time: (int) The dead threshold time.
            process_pool_queue_size: (int) The process pool queue size.
            all_ready: (bool) Whether to wait for all consumers to be ready.
            device: (str) The device of the model, CPU(cpu) or GPU(cuda:x).
        """
    def task(self, args) -> None:
        """
        Task function. This function will be called by the CProcessor.
        Args:
            args: The arguments.

        Returns: None

        """
    def _start(self) -> None:
        """
        Start the video streamers' node.
        Returns: None

        """

class ImageStreamer(Streamer):
    block_mode: Incomplete
    once_mode: Incomplete
    video_fps: Incomplete
    video_width: Incomplete
    video_height: Incomplete
    play_interval: Incomplete
    output_wh: Incomplete
    padding: Incomplete
    ground_image_path: Incomplete
    _process_frame_interval: Incomplete
    cmask: Incomplete
    def __init__(self, nodeName, stream, queue_size, output_wh=(1920, 1080), padding: bool = False, play_interval: float = 0.04, process_frame_interval: int = 0, ground_image_path=None, block_mode: bool = False, once_mode: bool = True, all_ready: bool = True, device: str = 'cuda:0') -> None:
        '''
        ImageStreamer is the base class for the image streamer node.
        Args:
            nodeName: (str) The name of the node.
            stream: (str) The stream is folder or file path or Queue object(python multiprocessing.Queue).
                    Examples:
                    folder path: "./test_images".
                    file path: "./test_images/test.jpg".
                    Queue format: {"images_path": ["./test_images/test.jpg", "./test_images/test1.png", ...], "images_info"[{...}, {...}, ...]} or {"images_array": [np.ndarray, np.ndarray, ...], "images_info"[{...}, {...}, ...]}.
            queue_size: (int) The size of the queue.
            output_wh: (tuple) The output width and height.
            padding: (bool) Whether to padding the image.
            play_interval: (float) The play interval.
            process_frame_interval: (int) The interval of processing frames.
            ground_image_path: (str) The ground image path.
            block_mode: (bool) Whether to block the frame.
            once_mode: (bool) Whether to run once.
            all_ready: (bool) Whether to wait for all consumers to be ready.
            device: (str) The device of the model, CPU(cpu) or GPU(cuda:x).
        Examples:
            1.
            ImageStreamer("image_streamer", "./test_images", 32, output_wh=(1920, 1080), padding=True, play_interval=0.04, block_mode=False, once_mode=True, all_ready=True, device="cuda:0")
            2.
            ImageStreamer("image_streamer", "./test_images/test.jpg", 32, output_wh=(1920, 1080), padding=True, play_interval=0.04, block_mode=False, once_mode=True, all_ready=True, device="cuda:0")
            3.
            queue = multiprocessing.Queue(64)
            ImageStreamer("image_streamer", queue, 32, output_wh=(1920, 1080), padding=True, play_interval=0.04, block_mode=False, once_mode=False, all_ready=True, device="cuda:0")

        '''
    def get_one_image(self):
        """
        Get one image from the stream.
        Returns: (np.ndarray) The image.

        """
    def _start(self) -> None:
        """
        Start the streamer node.
        Returns: None

        """

class MCPStreamer(Streamer):
    show_chinese: Incomplete
    show_args: Incomplete
    block_mode: Incomplete
    mcp_transport: Incomplete
    mcp_host: Incomplete
    mcp_port: Incomplete
    mcp_tool_name: Incomplete
    video_fps: Incomplete
    video_width: Incomplete
    video_height: Incomplete
    _process_frame_interval: Incomplete
    cevents: Incomplete
    cmask: Incomplete
    Image: Incomplete
    PILImage: Incomplete
    cmcp: Incomplete
    def __init__(self, nodeName, queue_size, mcp_tool_name, mcp_transport: str = 'streamable-http', mcp_host: str = '0.0.0.0', mcp_port: int = 19966, process_frame_interval: int = 0, image_show_wh=(640, 640), block_mode: bool = False, all_ready: bool = True, device: str = 'cuda:0') -> None:
        """
        ImageStreamer is the base class for the image streamer node.
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The size of the queue.
            mcp_tool_name: (str) The name of the mcp tool. Clearly describe the capabilities of this MCP.
            mcp_transport: (str) The transport of the mcp server. Just support streamable-http.
            mcp_host: (str) The host of the mcp server.
            mcp_port: (int) The port of the mcp server.
            process_frame_interval: (int) The interval of processing frames.
            image_show_wh: (tuple) The size of the image to show.
            block_mode: (bool) Whether to block the frame.
            all_ready: (bool) Whether to wait for all consumers to be ready.
            device: (str) The device of the model, CPU(cpu) or GPU(cuda:x).
        Examples:

        """
    def mcp_report(self, data) -> None:
        """
        MCP report with event.
        Args:
            data: (tuple) The data.
        """
    def get_one_image(self):
        """
        Get one image from the stream.
        Returns: (np.ndarray) The image.

        """
    async def inference(self, image_base64: str = '', image_path: str = '', image_url: str = '', result_inference_image: bool = True, result_embedding: bool = True):
        '''
        Inference the image. The reasoning results include target detection/classification/segmentation/face recognition/key points/OCR, etc.
        The image can be provided in one of the following ways:
        - image_base64: (str) The base64 of the image.
        - image_path: (str) The local file path of the image. e.g. "./test.jpg"
        - image_url: (str) The url of the image. e.g. "http://****.png"
        - result_inference_image: (bool) Whether to return the inference image. And show bboxes on the image.
        - result_embedding: (bool) Whether to return the embedding(512/768).

        Args:
            image_base64: (str) The base64 of the image.
            image_path: (str) The path of the image.
            image_url: (str) The url of the image.
            result_inference_image: (bool) Whether to return the inference image. And show bboxes on result image.
            result_embedding: (bool) Whether to return the embedding(512/768).

        Returns:
            Image: The image base64 string.
            dict: The inference result. The format is:
                e.g: {
                "box_coord": [x1, y1, x2, y2],
                "box_polygon_coord": [x1, y1, x2, y1, x2, y2, x1, y2],
                "box_angle": 0,
                "box_score": 0.9,
                "box_class": 0,
                "box_class_name": "person",
                "box_img": [height, width, channel], # default is None
                "box_key_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], ...], # default is None
                "box_key_point_scores": [0.9, 0.9, 0.9, 0.9, 0.9, ...], # default is None
                "box_key_point_names": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", ...], # default is None
                "box_mask": [height, width], # default is None
                "box_embedding": [512], # 512/768 embedding vector, default is None
                "box_embedding_name": "person", # default is None
                "box_embedding_score": 0.9, # default is None
                "box_text": "this is a person, she is 18 years old", # default is None
                "classification": {
                                    "names": ["person", "girl"],
                                    "scores": [0.9, 0.8],
                                    "classes": [1, 2]
                                }, # default is None
                "person": {
                            "person_box_coord": [x1, y1, x2, y2],
                            "person_box_score": 0.9,
                            "person_embedding": [512],
                            "person_score": 0.9,
                            "person_name": "Lily",
                            "face_box_coord": [x1, y1, x2, y2],
                            "face_box_score": 0.9,
                            "face_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], ...],
                            "face_embedding": [512],
                            "face_score": 0.9,
                            "face_name": "Lily\'s face",
                            "face_img": [height, width, channel],
                            "track": {
                                        "track_id": 1,
                            }, # default is None
                        }, # default is None
                "track": {"track_id": 1}, # default is None
        '''
    def _start(self) -> None:
        """
        Start the streamer node.

        Returns: None
        """
