from _typeshed import Incomplete
from cpipe.module.cdata import Box as Box, Person as Person
from cpipe.module.cinferencer import CEmbedding as CEmbedding
from cpipe.module.dataprocessing import embedding_preprocess as embedding_preprocess, load_data as load_data
from cpipe.module.node import Node as Node

class PPLCNet(CEmbedding):
    preprocessor: Incomplete
    mean: Incomplete
    std: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names=(), max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        PPLCNet is a class for PPLCNet model.

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
            *args:  
            **kwargs:
        '''
    def init_mean_std(self) -> None: ...
    def preprocess(self, frames, *args, **kwargs): ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None: ...
