from _typeshed import Incomplete
from cpipe.module.cinferencer import CFace as CFace
from cpipe.module.dataprocessing import load_data as load_data
from cpipe.module.inferenceengine import Cmodel as Cmodel
from cpipe.module.node import Node as Node

arcface_dst: Incomplete

def estimate_norm(lmk, image_size: int = 112, mode: str = 'arcface'):
    """
    Estimate the normalization.
    Args:
        lmk: (np.ndarray) The landmark.
        image_size: (int) The image size.
        mode: (str) The mode.

    Returns: The normalization.

    """
def norm_crop(img, landmark, image_size: int = 112, mode: str = 'arcface'):
    """
    Normalize the image.
    Args:
        img: (np.ndarray) The image.
        landmark: (np.ndarray) The landmark.
        image_size: (int) The image size.
        mode: (str) The mode.

    Returns: The normalized image.

    """
def preprocessor_fun(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for classification
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """

class Adaface(CFace):
    input_mean: float
    input_std: Incomplete
    bgr2gray_kernel: Incomplete
    inputs_nodeNames: Incomplete
    face_quality_model_path: Incomplete
    face_quality_model: Incomplete
    face_quality_thresh: Incomplete
    preprocessor: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize=(3, 112, 112), class_names=(), face_quality_model_path=None, face_quality_thresh: float = 0.5, max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=['output_layer', 'onnx::Div_1386'], save_face_image: bool = False, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        Adaface is a class for Adaface.
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
            save_face_image: (bool) The save face image to cdata flag.
            gray_mode: (bool) The gray mode flag.
            *args:
            **kwargs:
        '''
    def before_start(self) -> None: ...
    @staticmethod
    def matching(embeddings, face_embeddings):
        """
        Matching function.
        Args:
            embeddings: The embeddings.
            face_library: The face library.

        Returns: The matching result.

        """
    model: Incomplete
    def get_embedding(self, imgs):
        """
        Get the embedding of the image.
        Args:
            imgs: (torch.Tensor) The image tensor.

        Returns:

        """
    def get_feat(self, imgs):
        """
        Get the feature of the image
        Args:
            imgs: (torch.Tensor) The image tensor.

        Returns: The feature of the image.

        """
    def infer(self, inputs, *args, **kwargs): ...
    def preprocess(self, inputs, *args, **kwargs): ...
