from _typeshed import Incomplete
from cpipe.module.cdata import Box as Box
from cpipe.module.dataprocessing import load_data as load_data
from cpipe.module.inferenceengine import InferenceEngine as InferenceEngine

class MoveNet(InferenceEngine):
    feature_size: Incomplete
    center_weight: Incomplete
    range_weight_x: Incomplete
    range_weight_y: Incomplete
    hm_th: Incomplete
    num_joints: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, center_weight_path, num_joints, max_batch_size: int = 1, hm_th: float = 0.1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        """
        MoveNet is a class for MoveNet model.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 256, 256]
            class_names: (list) The class names.
            center_weight_path: (str) The path of the center weight.
            num_joints: (int) The number of joints.
            max_batch_size: (int) The max batch size.
            hm_th: (float) The heatmap threshold.
            warmup: (bool) The warmup flag.
            device: (str) The device.
            threading_num: (int) The threading number.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input names.
            output_names: (list) The output names.
            gray_mode: (bool) Whether to use gray mode.
        """
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None: ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
    @staticmethod
    def pre_process(self, raw_bgr_image, num): ...
    def preprocess(self, frames, *args, **kwargs): ...
    def infer(self, inputs, *args, **kwargs): ...
    def maxPoint(self, heatmap, center: bool = True):
        """
        Get the max point of heatmap
        Args:
            heatmap: The heatmap.
            center: The center flag.

        Returns: The x, y of the max point.

        """
    def postprocess(self, inputs, *args, **kwargs):
        """
        Post process method.
        Args:
            inputs: The inputs.
            *args:
            **kwargs:

        Returns:

        """

class MoveNetPersonPose(InferenceEngine):
    feature_size: Incomplete
    range_weight_x: Incomplete
    range_weight_y: Incomplete
    num_joints: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, num_joints, max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        """
        MoveNet is a class for MoveNet model.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 256, 256]
            class_names: (list) The class names.
            num_joints: (int) The number of joints.
            max_batch_size: (int) The max batch size.
            warmup: (bool) The warmup flag.
            device: (str) The device.
            threading_num: (int) The threading number.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input names.
            output_names: (list) The output names.
            gray_mode: (bool) Whether to use gray mode.
        """
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None: ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
    @staticmethod
    def pre_process(self, raw_bgr_image, num): ...
    def preprocess(self, frames, *args, **kwargs): ...
    def infer(self, inputs, *args, **kwargs): ...
    def postprocess(self, inputs, *args, **kwargs):
        """
        Post process method.
        Args:
            inputs: The inputs.
            *args:
            **kwargs:

        Returns:

        """
