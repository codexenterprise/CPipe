from cpipe.config.config import *
from _typeshed import Incomplete
from cpipe.module.clogger import CLogger as CLogger
from cpipe.module.security import AESHelper as AESHelper, Security as Security, cp_m_i as cp_m_i, cp_m_p as cp_m_p
from cpipe.module.utils import cudaSetDevice as cudaSetDevice

class Cmodel:
    TRT_MODEL_LIST: Incomplete
    ONNX_MODEL_LIST: Incomplete
    TORCH_MODEL_LIST: Incomplete
    OM_MODEL_LIST: Incomplete
    RKNN_MODEL_LIST: Incomplete
    MODEL_TYPE_TRT: str
    MODEL_TYPE_ONNX: str
    MODEL_TYPE_TORCH: str
    MODEL_TYPE_RKNN: str
    MODEL_TYPE_OM: str
    logger: Incomplete
    nodeName: Incomplete
    modelPath: Incomplete
    modelType: Incomplete
    _warmup: Incomplete
    _inputSize: Incomplete
    _device: Incomplete
    _inputNames: Incomplete
    _outputNames: Incomplete
    _max_batch_size: Incomplete
    __model: Incomplete
    def __init__(self, nodeName, modelPath, inputSize, max_batch_size: int = 1, device: str = 'cuda:0', warmup: bool = True, inputNames=None, outputNames=None) -> None:
        '''
        Initialize a CPipe inference model instance.
        Args:
            nodeName: Delete the mask of the node.
            modelPath: The path of the model file.
            inputSize: The size of the input image.
            max_batch_size: The maximum batch size of the model.
            device: The device of the model. Default is "cuda:0".
            warmup: Whether to warm up the model.
            inputNames: The input names of the model.
            outputNames: The output names of the model.

        '''
    @classmethod
    def get_encryption_model(cls, modelPath):
        """
        Get the encryption model.
        Args:
            modelPath: The path of the model file.

        Returns: The encryption model.
        """
    @classmethod
    def onnx2tensorrt(cls, modelPath, max_batch_size: int = 1, input_height=None, input_width=None, min_shapes=None, opt_shapes=None, max_shapes=None, input_names=None, fp16_mode: bool = True, int8_mode: bool = False):
        '''
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
        example:
            # no batch size
            from cpipe.module.cmodel import Cmodel
            Cmodel.onnx2tensorrt("./yolov10n.onnx", [(1, 3, 640, 640)], [(1, 3, 640, 640)], [(1, 3, 640, 640)], input_names=["images"], fp16_mode=True, int8_mode=False)
            # with dynamic batch size
            from cpipe.module.cmodel import Cmodel
            Cmodel.onnx2tensorrt("./yolov10n.onnx", [(1, 3, 640, 640)], (1, 3, 640, 640), [(16, 3, 640, 640)], input_names=["images"], fp16_mode=True, int8_mode=False)
        Returns:
            True: success
            False: failed
        '''
    @staticmethod
    def get_model_type(modelPath):
        """
        Get the model type from the model path.
        Args:
            modelPath: The path of the model file.

        Returns: The model type.

        """
    def _model(self, input_data):
        """
        The model inference function.
        Args:
            input_data: The input data.

        Returns: The model inference result.

        """
    def om_loadModel(self, modelPath, input_size, max_batch_size, inputNames=None, outputNames=None):
        """
        Load the OM model(HUAWEI Ascend).
        Args:
            modelPath: The path of the model file.
            input_size: The size of the input image.
            max_batch_size: The maximum batch size of the model.
            inputNames: The input names of the model.
            outputNames: The output names of the model.

        Returns: The model, input names, output names.

        """
    def rknn_loadModel(self, modelPath, input_size, max_batch_size, device: int = 0, inputNames=None, outputNames=None):
        """
        Load the RKNN model.
        Args:
            modelPath: The path of the model file.
            input_size: The size of the input image.
            max_batch_size: The maximum batch size of the model.
            device: The device of the model. eg: RKNNLite.NPU_CORE_AUTO
                        NPU_CORE_AUTO  = 0                                   # default, run on NPU core randomly.
                        NPU_CORE_0     = 1                                   # run on NPU core 0.
                        NPU_CORE_1     = 2                                   # run on NPU core 1.
                        NPU_CORE_2     = 4                                   # run on NPU core 2.
                        NPU_CORE_0_1   = 3                                   # run on NPU core 1 and core 2.
                        NPU_CORE_0_1_2 = 7                                   # run on NPU core 1 and core 2 and core 3.
                        NPU_CORE_ALL   = 0xffff                              # run on all NPU cores.
            inputNames: The input names of the model.
            outputNames: The output names of the model.

        Returns: The model, input names, output names.

        """
    def trt_loadModel(self, modelPath, input_size, max_batch_size, device, inputNames=None, outputNames=None):
        """
        Load the TensorRT model.
        Args:
            modelPath: The path of the model file.
            input_size: The size of the input image.
            max_batch_size: The maximum batch size of the model.
            device: The device of the model.
            inputNames: The input names of the model.
            outputNames: The output names of the model.

        Returns: The model, input names, output names.

        """
    def onnx_loadModel(self, modelPath, input_size, max_batch_size, device, inputNames=None, outputNames=None):
        """
        Load the ONNX model.
        Args:
            modelPath: The path of the model file.
            input_size: The size of the input image.
            max_batch_size: The maximum batch size of the model.
            device: The device of the model.
            inputNames: The input names of the model.
            outputNames: The output names of the model.

        Returns: The model, input names, output names.

        """
    def torch_loadModel(self, modelPath, input_size, max_batch_size, device):
        """
        Load the TorchScript model.
        Args:
            modelPath: The path of the model file.
            input_size: The size of the input image.
            max_batch_size: The maximum batch size of the model.
            device: The device of the model.

        Returns: The model, [], [].

        """
    def loadModel(self, modelPath) -> None:
        """
        Load the model.
        Args:
            modelPath: The path of the model file.

        Returns: None

        """
    def __call__(self, inputs, list_mode: bool = False):
        """
        Model inference function.
        Args:
            inputs: The input data.
            list_mode: Whether to return the result in list mode.

        Returns: The model inference result.

        """
