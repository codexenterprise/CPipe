
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
    


class CImage:
    pass


class CData:
    def __init__(self, createNodeName):
        """
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
        audio_texts: The audio data of the CData object. The format is {"stream_name1": "this is text", "stream_name2": "this is text", ...}.

        """
        self.createNodeName = createNodeName
        self.timestamp = 0

        self.json = {}
        self.images = {}
        self.states = {}
        self.shows = {}
        self.messages = {}
        self.bboxes = {}
        self.audio_texts = {}

    def get_bboxes(self):
        """
        Get the Box data of the CData object.
        Returns: The Box data of the CData object. The format is (["stream_name", "stream_name", ...], [[Box, Box, Box, ...], [Box, Box, Box, ...], ...]).

        """
        return list(self.bboxes.keys()), list(self.bboxes.values())