import time
from typing import Optional
import numpy as np

from cpipe.config.config import SHARE_MEMORY_MODE, CPIPE_BLOCKING_MODE

class LLMInfo:
    def __init__(self, system_prompt=None, user_prompt=None, file_path=None, response=None):
        """
        A LLMInfo object in CPipe corresponds to the LLM model data.
        Attributes:
        system_prompt: The system prompt of the LLM model.
        user_prompt: The user prompt of the LLM model.
        file_path: The file path of the image(can be a list of file paths or a single file path or numpy.ndarray or list of numpy.ndarray).
        response: The response of the LLM model. The default value is None.
        """
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.file_path = file_path
        self.response = response
    
    def to_json(self):
        """
        Convert the VLM object to a JSON object.
        Returns: The JSON object.
        """
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "file_path": self.file_path,
            "response": self.response
        }

class Tracker:
    def __init__(self):
        """
        A Tracker object in CPipe corresponds to the tracking information of a target. A Tracker object contains the tracking ID and tracking history of the target.

        Attributes:
        track_id: The tracking ID of the target. The default value is 0.
        track_history: The tracking history of the target.
        """
        self.track_id = 0
        self.track_history = []
        self.velocity = None
    
    def to_json(self):
        """
        Convert the Tracker object to a JSON object.
        Returns: The JSON object.
        """
        return {
            "track_id": int(self.track_id),
            "velocity": self.velocity.tolist() if self.velocity is not None else None
            # "track_history": list(self.track_history),
        }


class Person:
    def __init__(self):
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
        person_attribute: The attribute of the person. The default value is None. eg: Hat/Glasses/ShortSleeve/LongSleeve/UpperStride/UpperLogo/UpperPlaid/UpperSplice/LowerStripe/LowerPattern/LongCoat/Trousers/Shorts/Skirt&Dress/boots/HandBag/ShoulderBag/Backpack/HoldObjectsInFront/AgeOver60/Age18-60/AgeLess18/Female/Front/Side/Back
        track: The tracking information of the person. It is an instance of the Tracker class. The default value is None.

        """
        self.person_box_coord = []
        self.person_box_score = 0.

        self.person_embedding = None
        self.person_score = 0.
        self.person_name = ""

        self.face_box_coord = []
        self.face_box_score = 0.

        self.face_key_points = []

        self.face_embedding = None
        self.face_score = 0.
        self.face_name = ""
        self.face_img = None

        self.person_attribute = []

        self.track: Optional[Tracker] = None
    
    def to_json(self, result_embedding=True):
        """
        Convert the Person object to a JSON object.
        Args:
            result_embedding: (bool) Whether to return the embedding(512/768).
        Returns: The JSON object.
        """
        json_dict = {
            "person_box_coord": np.array(self.person_box_coord, dtype=np.int32).tolist(),
            "person_box_score": round(float(self.person_box_score), 3),
            "person_embedding": np.array(self.person_embedding).tolist() if self.person_embedding is not None and result_embedding else None,
            "person_score": round(float(self.person_score),3),
            "person_name": self.person_name,
            "face_box_coord": np.array(self.face_box_coord, dtype=np.int32).tolist(),
            "face_box_score": round(float(self.face_box_score),3),
            "face_key_points": np.array(self.face_key_points, dtype=np.int32).tolist(),
            "face_embedding": np.array(self.face_embedding).tolist() if self.face_embedding is not None and result_embedding else None,
            "face_score": round(float(self.face_score),3),
            "face_name": self.face_name,
            "face_img": self.face_img.tolist() if self.face_img is not None else None,
            "track": self.track.to_json() if self.track else None,
        }

        return json_dict


class Classification:
    def __init__(self):
        """
        A Classification object in CPipe corresponds to the classification information of a target.

        Attributes:
        names: The names of the target. The default value is an empty list.
        scores: The confidence scores of the target. The default value is an empty list.
        classes: The class IDs of the target. The default value is an empty list.

        """
        self.names = []
        self.scores = []
        self.classes = []
    
    def to_json(self):
        """
        Convert the Classification object to a JSON object.
        Returns: The JSON object.
        """
        return {
            "names": self.names,
            "scores": np.array(self.scores).tolist(),
            "classes": np.array(self.classes).tolist(),
        }


class Box:

    def __init__(self):
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

        self.box_coord = []
        self.box_polygon_coord = []
        self.box_angle = 0.
        self.box_score = 0.
        self.box_class = 0
        self.box_class_name = ""
        self.box_img = None
        self.box_key_points = None
        self.box_key_point_scores = None
        self.box_key_point_names = None
        self.box_mask = None
        self.box_embedding = None
        self.box_embedding_name = None
        self.box_embedding_score = None
        self.box_text = None

        self.classification: Optional[Classification] = None

        self.person: Optional[Person] = None
        self.track: Optional[Tracker] = None
        self.llm_info: Optional[LLMInfo] = None
    
    def to_json(self, result_embedding=True):
        """
        Convert the Box object to a JSON object.
        Args:
            result_embedding: (bool) Whether to return the embedding(512/768).
        Returns: The JSON object.
        """
        json_dict = {
            "box_coord": np.array(self.box_coord, dtype=np.int32).tolist(),
            "box_polygon_coord": np.array(self.box_polygon_coord, dtype=np.int32).tolist(),
            "box_angle": round(float(self.box_angle),3),
            "box_score": round(float(self.box_score),3),
            "box_class": int(self.box_class),
            "box_class_name": self.box_class_name,
            "box_img": self.box_img.tolist() if self.box_img is not None else None,
            "box_key_points": np.array(self.box_key_points).tolist() if self.box_key_points is not None else None,
            "box_key_point_scores": np.array(self.box_key_point_scores).tolist() if self.box_key_point_scores is not None else None,
            "box_key_point_names": self.box_key_point_names,
            "box_mask": np.array(self.box_mask, dtype=np.uint8).tolist() if self.box_mask is not None else None,
            "box_embedding": np.array(self.box_embedding).tolist() if self.box_embedding is not None and result_embedding else None,
            "box_embedding_name": self.box_embedding_name,
            "box_embedding_score": round(float(self.box_embedding_score),3) if self.box_embedding_score is not None else None,
            "box_text": self.box_text,
            "classification": self.classification.to_json() if self.classification else None,
            "person": self.person.to_json(result_embedding) if self.person else None,
            "track": self.track.to_json() if self.track else None,
            "llm_info": self.llm_info.to_json() if self.llm_info else None,
        }

        return json_dict


class CImage:
    def __init__(self, streamer_name=None):
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
        self.__share_m_flag = SHARE_MEMORY_MODE
        self.__image = None
        self.__image_timestamp = -1
        self.__share_m_idx = 0
        self.frame_total = 0
        self.frame_current_idx = 0
        self.image_marker = None
        self.streamer_name = streamer_name
        self.cbuffer_main_name = None
        self.cbuffer_sub_name = None
        self.info = {}
    
    def copy_image_info(self, cimage):
        """
        Copy the image info from the cimage object.
        Args:
            cimage: The cimage object to copy the image info from.
        """
        self.image_marker = cimage.image_marker
        self.streamer_name = cimage.streamer_name
        self.info = cimage.info
        self.frame_total = cimage.frame_total
        self.frame_current_idx = cimage.frame_current_idx
        self.image_timestamp = cimage.image_timestamp

    def share_m_idx(self):
        """
        Get the shared memory index of the image data.
        Returns: The shared memory index of the image data.

        """
        return self.__share_m_idx

    def get_image(self, node_buffers, shape=None):
        """
        Get the image data.
        Args:
            node_buffers: The node buffers.
            shape: The shape of the image data.

        Returns: The image data.

        """
        if self.__share_m_flag:
            cbuffer = node_buffers[self.cbuffer_main_name][self.cbuffer_sub_name]
            return cbuffer.data(self.__share_m_idx, shape=shape)
        else:
            if self.__image is not None:
                return self.__image
            else:
                raise ValueError("Image is None")

    def free(self, node_buffers, consumer_name):
        """
        Free the image data.
        Args:
            node_buffers: The node buffers.
            consumer_name: The name of the consumer.
        """
        if CPIPE_BLOCKING_MODE:
            if self.__share_m_flag:
                cbuffer = node_buffers[self.cbuffer_main_name][self.cbuffer_sub_name]
                cbuffer.free(self.__share_m_idx, consumer_name)
    
    def malloc(self, cbuffer, cbuffer_sub_name, value=None, shape=None):
        """
        Malloc the image data.
        Args:
            cbuffer: The shared memory buffer object.
            cbuffer_sub_name: The sub name of the shared memory buffer.
            value: The image data.
            shape: The shape of the image data. [height, width, channel]
        """
        if value is None and shape is None:
            raise ValueError("value and shape cannot be None at the same time")
        
        if self.__share_m_flag:
            share_memory, self.__share_m_idx = cbuffer.malloc(value, shape)
            self.cbuffer_main_name = cbuffer.name
            self.cbuffer_sub_name = cbuffer_sub_name
            return share_memory
        else:
            self.__image = value
            self.__image_timestamp = time.time()
            return self.__image
    
    @property
    def image_timestamp(self):
        """
        Get the timestamp of the image data.
        Returns: The timestamp of the image data.

        """
        return self.__image_timestamp

    @image_timestamp.setter
    def image_timestamp(self, value):
        """
        Set the timestamp of the image data.
        Args:
            value: The timestamp of the image data.

        Returns:

        """
        self.__image_timestamp = value


class CData:
    def __init__(self, create_node_name):
        """
        CData is the core data storage class of CPipe, which is used to store the output results of all detection, recognition, tracking, and other model and logic codes.
        Any image inference result and logic result should be converted into a Box object data and then stored in the CData object.

        Args:
        create_node_name: The name of the node that generates the CData object.

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
        audio_texts: The audio data of the CData object. The format is {"stream_name1": "this is text", "stream_name2": "this is text", ...}.
        llm_infos: The LLM data of the CData object. The format is {"stream_name": [LLMInfo, LLMInfo, LLMInfo, ...], "stream_name2": [LLMInfo, LLMInfo, LLMInfo, ...], ...}.

        """
        self.create_node_name = create_node_name
        self.timestamp = 0

        self.json = {}
        self.images = {}
        self.states = {}
        self.shows = {}
        self.messages = {}
        self.bboxes = {}
        self.audio_texts = {}
        self.llm_infos = {}
    
    def is_empty(self):
        """
        Check if the CData object is empty.
        Returns: True if the CData object is empty, False otherwise.
        """
        return len(self.bboxes) == 0 and len(self.images) == 0 and len(self.json) == 0 and len(self.states) == 0 and len(self.shows) == 0 and len(self.messages) == 0 and len(self.audio_texts) == 0

    def add_bboxes(self, streamer_name, boxes):
        """
        Add Box data to the CData object.
        Args:
            streamer_name: The streamer name corresponding to these Box data.
            boxes: The Box data. The format is [Box, Box, Box, ...].

        Returns: None

        """
        if streamer_name not in self.bboxes.keys():
            self.bboxes[streamer_name] = []
        if isinstance(boxes, list):
            self.bboxes[streamer_name].extend(boxes)
        else:
            self.bboxes[streamer_name].append(boxes)

    def get_bboxes(self):
        """
        Get the Box data of the CData object.
        Returns: The Box data of the CData object. The format is (["stream_name", "stream_name", ...], [[Box, Box, Box, ...], [Box, Box, Box, ...], ...]).

        """
        return list(self.bboxes.keys()), list(self.bboxes.values())

    def merge(self, cdata):
        """
        Merge the CData object.
        Args:
        cdata: The CData object to be merged.

        """
        self.timestamp = time.time()

        for one in cdata.bboxes.keys():
            self.add_bboxes(one, cdata.bboxes[one])

        self.images.update(cdata.images)

        self.json.update(cdata.json)
        self.states.update(cdata.states)
        self.shows.update(cdata.shows)
        self.messages.update(cdata.messages)
        self.audio_texts.update(cdata.audio_texts)
        self.llm_infos.update(cdata.llm_infos)

        return self
