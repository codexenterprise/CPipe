from _typeshed import Incomplete
from cpipe.module.cinferencer import CSemanticSegmentation as CSemanticSegmentation
from cpipe.module.dataprocessing import load_data as load_data
from cpipe.module.node import Node as Node

def mm_ss_preprocessor(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for classification
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """

class MMSemanticSegmentation(CSemanticSegmentation):
    mean: Incomplete
    stdinv: Incomplete
    preprocessor: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        MMSemanticSegmentation is a class for Semantic Segmentation model.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 224, 224]
            class_names: (list) The class names.
            max_batch_size: (int) The max batch size.
            warmup: (bool) The warmup flag.
            device: (str) The device. e.g. "cuda:0" or "cpu"
            threading_num: (int) The threading number.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input_names.
            output_names: (list) The output_names.
            gray_mode: (bool) The gray mode flag.
        '''
    def preprocess(self, frames, *args, **kwargs): ...
    def before_start(self) -> None: ...
    def postprocess(self, inputs, *args, **kwargs): ...
