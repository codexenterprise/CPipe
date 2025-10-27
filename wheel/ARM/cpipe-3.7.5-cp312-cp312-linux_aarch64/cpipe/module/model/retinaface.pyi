from _typeshed import Incomplete
from cpipe.module.cdata import Box as Box, Person as Person
from cpipe.module.cinferencer import CDetector as CDetector
from cpipe.module.cmodel import Cmodel as Cmodel
from cpipe.module.dataprocessing import distance2bbox as distance2bbox, distance2kps as distance2kps, load_data as load_data, preprocess_retinaface as preprocess_retinaface

class Retinaface(CDetector):
    face_recognition_width_limit: Incomplete
    face_recognition_height_limit: Incomplete
    face_wh_ratio: Incomplete
    face_filtering: Incomplete
    preprocessor: Incomplete
    save_person_image: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names=('face',), valid_class_names=None, max_batch_size: int = 1, face_recognition_width_limit: int = 38, face_recognition_height_limit: int = 38, face_filtering: bool = False, face_wh_ratio: float = 1.67, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, save_top_n_objects=None, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=('448', '471', '494', '451', '474', '497', '454', '477', '500'), save_person_image: bool = False, *args, **kwargs) -> None:
        '''
        RetinafaceTRT is a class for Retinaface TensorRT model.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 416, 416]
            class_names: (list) The class names.
            valid_class_names: (list) The valid class names.
            max_batch_size: (int) The max batch size.
            face_recognition_width_limit: (int) The face recognition width limit of pixel.
            face_recognition_height_limit: (int) The face recognition height limit of pixel.
            face_filtering: (bool) The face filtering flag.
            face_wh_ratio: (float) The face width and height ratio.
            warmup: (bool) The warmup flag.
            device: (str) The device. e.g. "cuda:0" or "cpu"
            threading_num: (int) The threading number.
            save_top_n_objects: (int) The save top n objects.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input_names.
            output_names: (list) The output_names.
            save_person_image: (bool) The save person image to cdata flag.
        '''
    det_thresh: float
    fmc: int
    _feat_stride_fpn: Incomplete
    _num_anchors: int
    use_kps: bool
    mean: float
    std: Incomplete
    _anchor_cache: Incomplete
    def _init_vars(self) -> None:
        """
        Initialize the variables.
        Returns: None

        """
    def preprocess(self, frames, *args, **kwargs): ...
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None: ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
    def infer(self, inputs, *args, **kwargs): ...
