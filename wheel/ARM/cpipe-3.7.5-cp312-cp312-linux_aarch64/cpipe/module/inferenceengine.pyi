from _typeshed import Incomplete
from cpipe.config.config import SHARE_MEMORY_MODE as SHARE_MEMORY_MODE
from cpipe.module.cdata import CData as CData
from cpipe.module.cinferencehook import CHook as CHook
from cpipe.module.cmodel import Cmodel as Cmodel
from cpipe.module.node import Node as Node

class InferenceEngine(Node):
    _modelPath: Incomplete
    _class_names: Incomplete
    _max_batch_size: Incomplete
    _warmup: Incomplete
    _area_flag: Incomplete
    _area_info: Incomplete
    _inputNames: Incomplete
    _outputNames: Incomplete
    _gray_mode: Incomplete
    _hook_inputs: Incomplete
    _hook_outputs: Incomplete
    _threading_num: Incomplete
    _half: bool
    _inputSize: Incomplete
    preprocessor: Incomplete
    secondary_class_names: Incomplete
    dump_images: bool
    device: Incomplete
    queue_size: Incomplete
    model: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, input_names=None, output_names=None, secondary_class_names=None, gray_mode: bool = False, hook_inputs: CHook = None, hook_outputs: CHook = None) -> None:
        """
        InferenceEngine is the base class for the model inference node.
        Args:
            nodeName: nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The size of the queue.
            inputSize: (list) The size of the input image.
            class_names: (list) The class names.
            max_batch_size: (int) The maximum batch size.
            warmup: (bool) Whether to warm up the model.
            device: (str) The device of the model, CPU(cpu) or GPU(cuda:x).
            threading_num: (int) The number of preprocessing threads.
            area_flag: (bool) Whether to use the area mask.
            input_names: (list) The input names of the model.
            output_names: (list) The output names of the model.
            secondary_class_names: (list) The class names of the previous node that need to be processed in the two-stage mode.
            gray_mode: (bool) Whether to use gray mode.
            hook_inputs: (function) The hook function of the inputs(Perform some logical operations on the input frames before the algorithm preprocessing.).
            hook_outputs: (function) The hook function of the outputs(Perform some logical operations on the output data before the algorithm postprocessing.).
        """
    def update_mask(self, data) -> None:
        """
        Update the mask of the streamer.
        """
    @property
    def _device(self):
        """
        Get the device of the model.
        Returns: None

        """
    @staticmethod
    def onnx2tensorrt(modelPath, max_batch_size: int = 1, input_height=None, input_width=None, min_shapes=None, opt_shapes=None, max_shapes=None, input_names=None, fp16_mode: bool = True, int8_mode: bool = False) -> None:
        """
        ONNX to TensorRT 
        
        Args:
            modelPath: onnx/cpipe/codex model path
            max_batch_size: max batch size
            input_height: input height
            input_width: input width
            min_shapes: min input shape
            opt_shapes: opt input shape
            max_shapes: max input shape
            input_names: input names
            fp16_mode: use fp16 mode
            int8_mode: use int8 mode

        """
    def half(self) -> None:
        """
        Set the model to half precision.
        Returns: None

        """
    def set_threading_num(self, num) -> None:
        """
        Set the number of preprocessing threads
        Args:
            num: preprocessing threading num

        Returns: None

        """
    def preprocess(self, inputs, *args, **kwargs):
        """
        Model input data preprocessing method.
        Args:
            inputs: input data
            *args:
            **kwargs:

        Returns: input data

        """
    def infer(self, inputs, *args, **kwargs):
        """
        Model inference method.
        Args:
            inputs: input data
            *args:
            **kwargs:

        Returns: model output data

        """
    def postprocess(self, inputs, *args, **kwargs):
        """
        Model output data post-processing method.
        Args:
            inputs: input data
            *args:
            **kwargs:

        Returns: output data

        """
    def forward(self, inputs, *args, **kwargs):
        """
        Model inference logic processing.
        Args:
            inputs: input data
            *args:
            **kwargs:

        Returns: output data

        """
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None:
        """
        The result processing function of the model inference (or logic) of the current node in the one-stage mode is completed,
         and the result is processed and saved to the CData object.
        Args:
            pred: model output.
            new_cdata: The CData object generated by the current node.
            frames: The original image of each batch corresponding to the inference.
            streamer_names: The stream name corresponding to each batch.
        Returns: None

        """
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None:
        """
        The result processing function of the model inference (or logic) of the current node in the two-stage mode is completed,
         and the result is processed and saved to the CData object.
        Args:
            pred: model output.
            new_cdata: The CData object generated by the current node.
            streamer_names: The stream name corresponding to each batch.
            box_idxes: The index of the Box in det_boxes corresponding to each batch.
            boxes: A list of all Box objects.
        Returns: None

        """
    def _loadModel(self, modelPath=None) -> None:
        """
        Load the model of the current node, if modelPath is None, the model is not loaded.
        Args:
            modelPath: model path

        Returns: None

        """
    def load_model(self) -> None:
        """
        Load the model of the current node.
        Args:

        Returns: None

        """
    def get_streamer_area(self, need_update_mask_map: bool = False) -> None:
        """
        Initialize the cmask data corresponding to the current node for subsequent model inference.
        Returns: None

        """
    def before_start(self) -> None:
        """
        This method can be used to customize the initialization operation of the current node, which will be executed before the node is ready.
        Returns: None

        """
    def __call__(self, inputs, return_cdata_format: bool = False, frames_stream_names=(), *args, **kwargs):
        """
        Model inference method.
        Args:
            inputs: input data
            frames_stream_names: The stream name corresponding to each batch. e.g. ['stream1', 'stream2']
            return_cdata_format: return cdata format
            *args:
            **kwargs:

        Returns: output data

        """
    def _start(self) -> None:
        """
        The entry function for the current node (process) to run in one-stage mode, all the logic processing of the current node is completed here.
        Returns: None

        """
    def _start_secondary(self) -> None:
        """
        The entry function for the current node (process) to run in two-stage mode, all the logic processing of the current node is completed here.
        Returns: None

        """
