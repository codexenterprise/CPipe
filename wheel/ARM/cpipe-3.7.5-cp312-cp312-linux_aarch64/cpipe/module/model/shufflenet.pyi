from _typeshed import Incomplete
from cpipe.module.cinferencer import CClassifier as CClassifier
from cpipe.module.dataprocessing import class_preprocess as class_preprocess, load_data as load_data, mm_class_preprocess as mm_class_preprocess

class MMShuffleNet(CClassifier):
    preprocessor: Incomplete
    mean: Incomplete
    std: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, max_batch_size: int = 1, conf_thres: float = 0.25, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        MMShuffleNet is a class for ShuffleNet model.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 224, 224]
            class_names: (list) The class names.
            max_batch_size: (int) The max batch size.
            conf_thres: (float) The confidence threshold.
            warmup: (bool) The warmup flag.
            device: (str) The device. e.g. "cuda:0" or "cpu"
            threading_num: (int) The threading number.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input_names.
            output_names: (list) The output_names.
            gray_mode: (bool) Whether to use gray mode.
        '''
    def preprocess(self, frames, *args, **kwargs):
        """
        The preprocess function of the model.
        Args:
            frames: The frames.
            *args: frames_stream_names. eg ['stream1', 'stream2']
            **kwargs: The keyword arguments.

        Returns: The batch_imgs, frames, batch_size

        """
    def infer(self, inputs, *args, **kwargs):
        """
        The infer function of the model
        Args:
            inputs: inputs[0]: batch_imgs, inputs[1]: original image, inputs[2]: batch_size
            *args:
            **kwargs:

        Returns: The result of the model.

        """

class ShuffleNet(CClassifier):
    preprocessor: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, max_batch_size: int = 1, conf_thres: float = 0.25, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        ShuffleNet is a class for Shuffle
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 224, 224]
            class_names: (list) The class names.
            max_batch_size: (int) The max batch size.
            conf_thres: (float) The confidence threshold.
            warmup: (bool) The warmup flag.
            device: (str) The device. e.g. "cuda:0" or "cpu"
            threading_num: (int) The threading number.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input_names.
            output_names: (list) The output_names.
            gray_mode: (bool) Whether to use gray mode.
            need_softmax: (bool) Whether to use softmax for post-processing. Default is True.
        '''
    def infer(self, inputs, *args, **kwargs):
        """
        The infer function of the model
        Args:
            inputs: inputs[0]: batch_imgs, inputs[1]: original image, inputs[2]: batch_size
            *args:
            **kwargs:

        Returns: The result of the model.

        """
