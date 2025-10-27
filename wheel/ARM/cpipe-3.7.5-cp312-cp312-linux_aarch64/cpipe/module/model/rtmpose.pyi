from _typeshed import Incomplete
from cpipe.module.cdata import Box as Box
from cpipe.module.dataprocessing import load_data as load_data
from cpipe.module.inferenceengine import InferenceEngine as InferenceEngine

def pad_resize(img, width, height):
    """
    Pad and resize the image.
    Args:
        img: The image.
        width: The width.
        height: The height.

    Returns: The padded and resized image.

    """
def preprocess(self, raw_bgr_image, num) -> None:
    """
    Preprocess the input image.
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """

class RTMPose(InferenceEngine):
    mean: Incomplete
    std: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=('keyPoints', 'scores'), gray_mode: bool = False, *args, **kwargs) -> None:
        """
        RTMPose is a class for RTMPose model.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The size of the queue.
            inputSize: (list) The size of the input image.
            class_names: (list) The class names.
            max_batch_size: (int) The maximum batch size.
            warmup: (bool) The warmup flag.
            device: (str) The device of the model, CPU(cpu) or GPU(cuda:x).
            threading_num: (int) The number of preprocessing threads.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The class names of the previous node that need to be processed in the two-stage mode.
            input_names: (list) The input names.
            output_names: (list) The output names.
            gray_mode: (bool) Whether to use gray mode.
        """
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None: ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
    def preprocess(self, frames, *args, **kwargs): ...
    def infer(self, inputs, *args, **kwargs): ...
    def postprocess(self, inputs, *args, **kwargs): ...
