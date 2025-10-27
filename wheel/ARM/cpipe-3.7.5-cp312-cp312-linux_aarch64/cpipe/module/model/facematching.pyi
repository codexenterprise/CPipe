from _typeshed import Incomplete
from cpipe.module.cinferencer import CEmbedding as CEmbedding
from cpipe.module.node import Node as Node

class FaceLibrary:
    face_embeddings: Incomplete
    face_names: Incomplete
    face_images_path: Incomplete
    face_model: Incomplete
    gray_mode: Incomplete
    def __init__(self, face_embeddings=None, face_names=None, face_images_path=None, face_model=None, gray_mode: bool = False) -> None:
        """
        FaceLibrary is a class for face library.
        Args:
            face_embeddings: (np.ndarray) The face embeddings.
            face_names: (list) The face names.
            face_images_path: (str) The path of the face images.
            face_model: (Adaface) The face model.
            gray_mode: (bool) The gray mode flag.

        """
    def get_features(self):
        """
        Get the features of the face images.
        Returns: The features of the face images.

        """

class FaceMatching(CEmbedding):
    faces_library: Incomplete
    face_images_path: Incomplete
    face_model: Incomplete
    matching_score_thresh: Incomplete
    matching: Incomplete
    to_cdata: Incomplete
    _start: Incomplete
    def __init__(self, nodeName, queue_size, matching, inputSize=(1, 512), matching_score_thresh: float = 0.25, faces_library=None, face_model=None, face_images_path=None, max_batch_size: int = 1024, device: str = 'cuda:0', threading_num: int = 4, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        FaceMatching is a class for face matching model.
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The queue size.
            matching: (function) The matching function.
            inputSize: (list) The input size. e.g. [1, 512]
            matching_score_thresh: (float) The matching score threshold.
            faces_library: (FaceLibrary) The face library.
            face_model: (Adaface) The face model.
            face_images_path: (str) The path of the face images.
            max_batch_size: (int) The max batch size.
            device: (str) The device. e.g. "cuda:0" or "cpu"
            threading_num: (int) The threading number.
            gray_mode: (bool) The gray mode flag.
            *args:
            **kwargs:
        '''
    def infer(self, inputs, *args, **kwargs): ...
    face_embeddings: Incomplete
    def before_start(self) -> None: ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
    def _start_secondary(self) -> None: ...
