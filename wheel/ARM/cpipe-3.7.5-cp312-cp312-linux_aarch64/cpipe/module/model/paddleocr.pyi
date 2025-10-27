from cpipe.module.cdata import Box as Box
from cpipe.module.dataprocessing import load_data as load_data
from cpipe.module.inferenceengine import InferenceEngine as InferenceEngine

class PaddleOCR(InferenceEngine):
    def __init__(self, nodeName, modelPath, queue_size, keys_txt_path, inputSize=(3, 48, 320), max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        PaddleOCR is a class for PaddleOCR model.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 48, 320]
            keys_txt_path: (str) The path of the keys txt file. e.g. "ppocr_keys_v1.txt"
            max_batch_size: (int) The max batch size.
            warmup: (bool) The warmup flag.
            device: (str) The device.
            threading_num: (int) The threading number.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input names.
            output_names: (list) The output names.
            gray_mode: (bool) Whether to use gray mode.
        '''
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None: ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
    @staticmethod
    def pre_process(self, raw_bgr_image, num) -> None: ...
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
