from cpipe.module.model.adaface import Adaface as Adaface

class Arcface(Adaface):
    def __init__(self, nodeName, modelPath, queue_size, inputSize=(3, 112, 112), class_names=(), face_score_thresh: float = 0.0, face_quality_model_path=None, face_quality_thresh: float = 0.5, max_batch_size: int = 1, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        '''
        Arcface is a class for Arcface.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 112, 112]
            class_names: (list) The class names.
            face_score_thresh: (float) The face score threshold.
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
            gray_mode: (bool) The gray mode flag.
        '''
    def get_feat(self, imgs):
        """
        Get the feature of the image
        Args:
            imgs: (torch.Tensor) The image tensor.

        Returns: The feature of the image.

        """
