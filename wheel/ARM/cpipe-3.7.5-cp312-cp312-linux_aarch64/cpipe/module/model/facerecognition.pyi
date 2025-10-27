from _typeshed import Incomplete
from cpipe.module.cdata import Box as Box, Person as Person
from cpipe.module.cmodel import Cmodel as Cmodel
from cpipe.module.model.adaface import Adaface as Adaface
from cpipe.module.model.facematching import FaceLibrary as FaceLibrary
from cpipe.module.node import Node as Node

class FaceRecognition(Adaface):
    faces_library: Incomplete
    face_images_path: Incomplete
    matching_score_thresh: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize=(3, 112, 112), class_names=(), face_quality_model_path=None, face_quality_thresh: float = 0.5, max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, faces_library=None, face_images_path=None, matching_score_thresh: float = 0.25, save_face_image: bool = True, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        FaceRecognition is a class for FaceRecognition model
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 112, 112]
            class_names: (list) The class names.
            face_quality_model_path: (str) The path of the face quality model.
            face_quality_thresh: (float) The face quality threshold.
            max_batch_size: (int) The max batch size.
            warmup: (bool) The warmup flag.
            device: (str) The device. e.g. "cuda:0" or "cpu"
            threading_num: (int) The threading number.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input_names.
            output_names: (list) The output_names.
            faces_library: (FaceLibrary) The face library.
            face_images_path: (str) The path of the face images.
            matching_score_thresh: (float) The matching score threshold.
            save_face_image: (bool) The save face image to cdata flag.
            gray_mode: (bool) The gray mode flag.
            *args:
            **kwargs:
        '''
    face_embeddings: Incomplete
    def before_start(self) -> None: ...
    def get_feat(self, imgs):
        """
        Get the feature of the image
        Args:
            imgs: (torch.Tensor) The image tensor.

        Returns: The feature of the image.

        """
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None: ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
