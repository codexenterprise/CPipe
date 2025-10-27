from _typeshed import Incomplete
from cpipe.config.config import CPIPE_BLOCKING_MODE as CPIPE_BLOCKING_MODE, SHARE_MEMORY_MODE as SHARE_MEMORY_MODE

class Tracker:
    track_id: int
    track_history: Incomplete
    def __init__(self) -> None:
        """
        A Tracker object in CPipe corresponds to the tracking information of a target. A Tracker object contains the tracking ID and tracking history of the target.

        Attributes:
        track_id: The tracking ID of the target. The default value is 0.
        track_history: The tracking history of the target.
        """
    def to_json(self):
        """
        Convert the Tracker object to a JSON object.
        Returns: The JSON object.
        """

class Person:
    person_box_coord: Incomplete
    person_box_score: float
    person_embedding: Incomplete
    person_score: float
    person_name: str
    face_box_coord: Incomplete
    face_box_score: float
    face_key_points: Incomplete
    face_embedding: Incomplete
    face_score: float
    face_name: str
    face_img: Incomplete
    track: Tracker | None
    def __init__(self) -> None:
        """
        A Person object in CPipe corresponds to the information of a person in the target.

        Attributes:
        person_box_coord: The coordinates of the person's bounding box. The format is [x1, y1, x2, y2].
        person_box_score: The confidence score of the person's bounding box. The value range is [0, 1].
        person_embedding: The feature vector of the person. The length of the feature vector is 512, is a numpy.ndarray.
        person_score: The confidence score of the person. The value range is [0, 1].
        person_name: The name of the person. The default value is an empty string.
        face_box_coord: The coordinates of the face's bounding box. The format is [x1, y1, x2, y2].
        face_box_score: The confidence score of the face's bounding box. The value range is [0, 1].
        face_points: The facial key points. The format is [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], ...].
        face_embedding: The feature vector of the face. The length of the feature vector is 512, is a numpy.ndarray.
        face_score: The confidence score of the face. The value range is [0, 1].
        face_name: The name of the face. The default value is an empty string.
        face_img: The image of the face. The format is [height, width, channel], is a numpy.ndarray.
        track: The tracking information of the person. It is an instance of the Tracker class. The default value is None.

        """
    def to_json(self, result_embedding: bool = True):
        """
        Convert the Person object to a JSON object.
        Args:
            result_embedding: (bool) Whether to return the embedding(512/768).
        Returns: The JSON object.
        """

class Classification:
    names: Incomplete
    scores: Incomplete
    classes: Incomplete
    def __init__(self) -> None:
        """
        A Classification object in CPipe corresponds to the classification information of a target.

        Attributes:
        names: The names of the target. The default value is an empty list.
        scores: The confidence scores of the target. The default value is an empty list.
        classes: The class IDs of the target. The default value is an empty list.

        """
    def to_json(self):
        """
        Convert the Classification object to a JSON object.
        Returns: The JSON object.
        """

class Box:
    box_coord: Incomplete
    box_polygon_coord: Incomplete
    box_angle: float
    box_score: float
    box_class: int
    box_class_name: str
    box_img: Incomplete
    box_key_points: Incomplete
    box_key_point_scores: Incomplete
    box_key_point_names: Incomplete
    box_mask: Incomplete
    box_embedding: Incomplete
    box_embedding_name: Incomplete
    box_embedding_score: Incomplete
    box_text: Incomplete
    classification: Classification | None
    person: Person | None
    track: Tracker | None
    def __init__(self) -> None:
        """
        Box class is a core data storage class of CPipe, which is used to store the output results of all detection, recognition, tracking, and other model and logic codes.
        Any image inference result and logic result should be converted into a Box object data and then stored in the CData object.

        Attributes:
        box_coord: The coordinates of the bounding box. The format is [x1, y1, x2, y2].
        box_polygon_coord: The coordinates of the polygon. The format is [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], ...].
        box_angle: The angle of the bounding box. The default value is 0.
        box_score: The confidence score of the bounding box. The value range is [0, 1].
        box_class: The class ID of the bounding box. The default value is 0.
        box_class_name: The name of the class. The default value is an empty string.
        box_img: The image of the bounding box. The format is [height, width, channel], is a numpy.ndarray.
        box_key_points: The key points of the bounding box. The format is [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], ...].
        box_key_point_scores: The confidence scores of the key points. The format is [score1, score2, score3, score4, score5, ...].
        box_key_point_names: The names of the key points. The format is [name1, name2, name3, name4, name5, ...].
        box_mask: The image mask of the bounding box. The format is a numpy.ndarray of [height, width]. Based on the coordinates of box_coord, applying box_mask to the original image can extract the target area corresponding to box_coord.
        box_embedding: The feature vector of the bounding box. The length of the feature vector is 512/768..., is a numpy.ndarray.
        box_embedding_name: The name of the feature vector. The default value is an empty string.
        box_embedding_score: The confidence score of the feature vector. The value range is [0, 1].
        box_text: The text(OCR result) of the bounding box. The default value is None.
        classification: The classification information of the target. It is an instance of the Classification class. The default value is None.
        person: The person information of the target. It is an instance of the Person class. The default value is None.
        track: The tracking information of the target. It is an instance of the Tracker class. The default value is None.
        """
    def to_json(self, result_embedding: bool = True):
        """
        Convert the Box object to a JSON object.
        Args:
            result_embedding: (bool) Whether to return the embedding(512/768).
        Returns: The JSON object.
        """

class CImage:
    __share_m_flag: Incomplete
    __image: Incomplete
    __image_timestamp: int
    __share_m_idx: int
    frame_total: int
    frame_current_idx: int
    image_marker: Incomplete
    streamer_name: Incomplete
    cbuffer_main_name: Incomplete
    cbuffer_sub_name: Incomplete
    info: Incomplete
    def __init__(self, streamer_name=None) -> None:
        """
        Used to store the image data of a Streamer in CPipe.

        Args:
            streamer_name: The streamer name of the Streamer. The default value is None. 

        Attributes:
        frame_total: Streamer total frame number. The default value is 0. Only the video stream read by Streamer has this attribute.
        frame_current_idx: The current frame index of the Streamer. The default value is 0. Only the video stream read by Streamer has this attribute.
        image_marker: The marker of the image. The default value is None. 
        streamer_name: The streamer name of the Streamer. The default value is None. 
        cbuffer_main_name: The main name of the shared memory buffer. The default value is None.
        cbuffer_sub_name: The sub name of the shared memory buffer. The default value is None.
        info: The information of the Streamer. The default value is an empty dictionary.

        """
    def copy_image_info(self, cimage) -> None:
        """
        Copy the image info from the cimage object.
        Args:
            cimage: The cimage object to copy the image info from.
        """
    def share_m_idx(self):
        """
        Get the shared memory index of the image data.
        Returns: The shared memory index of the image data.

        """
    def get_image(self, node_buffers, shape=None):
        """
        Get the image data.
        Args:
            node_buffers: The node buffers.
            shape: The shape of the image data.

        Returns: The image data.

        """
    def free(self, node_buffers, consumer_name) -> None:
        """
        Free the image data.
        Args:
            node_buffers: The node buffers.
            consumer_name: The name of the consumer.
        """
    def malloc(self, cbuffer, cbuffer_sub_name, value=None, shape=None):
        """
        Malloc the image data.
        Args:
            cbuffer: The shared memory buffer object.
            cbuffer_sub_name: The sub name of the shared memory buffer.
            value: The image data.
            shape: The shape of the image data. [height, width, channel]
        """
    @property
    def image_timestamp(self):
        """
        Get the timestamp of the image data.
        Returns: The timestamp of the image data.

        """
    @image_timestamp.setter
    def image_timestamp(self, value) -> None:
        """
        Set the timestamp of the image data.
        Args:
            value: The timestamp of the image data.

        Returns:

        """

class CData:
    createNodeName: Incomplete
    timestamp: int
    json: Incomplete
    images: Incomplete
    states: Incomplete
    shows: Incomplete
    messages: Incomplete
    bboxes: Incomplete
    def __init__(self, createNodeName) -> None:
        '''
        CData is the core data storage class of CPipe, which is used to store the output results of all detection, recognition, tracking, and other model and logic codes.
        Any image inference result and logic result should be converted into a Box object data and then stored in the CData object.

        Args:
        createNodeName: The name of the node that generates the CData object.

        Attributes:
        timestamp: The timestamp of the CData object.
        json: The JSON data of the CData object.
        images: The image data of the CData object.
        states: The special state data of the CData object. The format is {"any": any}.
        shows: The result of the algorithm display.
               eg:
               {
                   "stream_name": [
                       ("text", {"coord":np.array([996, 996], dtype=np.int32), "data": 996, "color":(0, 0, 255), "text_size": 40}),
                       ("line", {"coord":np.array([[996, 996], [996, 996]], dtype=np.int32), "color":(0, 0, 255), "thickness": 1}),
                       ("circle", {"coord": np.array([[996, 996], [996, 996]], dtype=np.int32), "color": (0, 255, 255), "radius": 1}),
                       ("polygon", {"coord":np.array([[700, 100], [1820, 980], [1000, 1030], [550, 800]], dtype=np.int32), "color":(0, 0, 255), "thickness": 1, "isclosed": True}),
                   ],
               }.
        messages: The message data of the CData object.
            eg:
               {
                   "stream_name": [
                       "msg1....",
                       "msg2....",
                       "msg3...."
                   ],
               }.
        bboxes: The Box data of the CData object. The format is {"stream_name": [Box, Box, Box, ...]}.

        '''
    def add_bboxes(self, streamer_name, boxes) -> None:
        """
        Add Box data to the CData object.
        Args:
            streamer_name: The streamer name corresponding to these Box data.
            boxes: The Box data. The format is [Box, Box, Box, ...].

        Returns: None

        """
    def get_bboxes(self):
        '''
        Get the Box data of the CData object.
        Returns: The Box data of the CData object. The format is (["stream_name", "stream_name", ...], [[Box, Box, Box, ...], [Box, Box, Box, ...], ...]).

        '''
    def merge(self, cdata):
        """
        Merge the CData object.
        Args:
        cdata: The CData object to be merged.

        """
