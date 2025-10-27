from _typeshed import Incomplete
from cpipe.module.cdata import Box as Box

class CHook:
    TYPE_INPUT: str
    TYPE_OUTPUT: str
    hook_type: Incomplete
    def __init__(self, hook_type: str) -> None:
        """
        CHook is the base class for all hooks.
        Args:
            hook_type: (str) CHook.TYPE_INPUT or CHook.TYPE_OUTPUT
        """
    def __call__(self, *args, **kwargs) -> None: ...

class HKI_CropImage(CHook):
    crop_factor: Incomplete
    def __init__(self, crop_factor) -> None:
        """
        Crop the image according to the crop factor.
        Args:
            crop_factor: (list) The crop factor of the input image. (h start(0~1), h end(0~1), w start(0~1), w end(0~1)).
        """
    def __call__(self, image, *args, **kwargs):
        """
        Crop the image according to the crop factor.
        Args:
            image: (numpy.ndarray) The image to be cropped.
            box: (Box) The box to be cropped. Just used in secondary mode.
            streamer_node_name: (Str) The stream node name.
            node: (Class) The InferenceEngine object.
        Returns:
            image: (numpy.ndarray) The cropped image. If box is None, dump the image.
        """

class HKI_DilateImage(CHook):
    dilate_factor: Incomplete
    def __init__(self, dilate_factor) -> None:
        """
        Dilate the image according to the dilate factor. Based on the box.
        Args:
            dilate_factor: (list) The dilate factor of the input image. (h(> 0), w(> 0)).
        """
    def __call__(self, image, *args, **kwargs):
        """
        Dilate the image according to the dilate factor.
        Args:
            image: (numpy.ndarray) The image to be dilated.
            box: (Box) The box to be dilated.
            streamer_node_name: (Str) The stream node name.
            node: (Class) The InferenceEngine object.
        Returns:
            image: (numpy.ndarray) The dilated image.
        """

class HKO_DumpClass(CHook):
    dump_class_names: Incomplete
    class_index: Incomplete
    def __init__(self, class_index, dump_class_names) -> None:
        """
        This is a demo of the hook function of the outputs.
        Args:
            class_index: (int) The index of the class in the output.
            dump_class_names: (list) The class names to be dumped.
        """
    def __call__(self, predictions, frames, model_class_names, *args, **kwargs):
        """
        This is a demo of the hook function of the outputs.
        Args:
            predictions: The output of the model.
            frames: The original image of each batch corresponding to the inference.
            model_class_names: The class names of the model.
            box_idxes: The index of the box in the output.
            boxes: The boxes of the output.
            node: (Class) The InferenceEngine object.
        Returns:
            None
        """

class HKO_ClassNamesThresholdFilter(CHook):
    class_names_threshold_dict: Incomplete
    class_names_list: Incomplete
    class_index: Incomplete
    confidence_index: Incomplete
    def __init__(self, class_index, confidence_index, class_names_threshold_dict: dict) -> None:
        '''
        This is a demo of the hook function of the outputs.
        Args:
            class_index: (int) The index of the class in the output.
            confidence_index: (int) The index of the confidence in the output.
            class_names_threshold_dict: (dict) The class names and the threshold. e.g. {"person": 0.5, "car": 0.3}
        '''
    def __call__(self, predictions, frames, model_class_names, *args, **kwargs):
        """
        This is a demo of the hook function of the outputs.
        Args:
            predictions: The output of the model.
            frames: The original image of each batch corresponding to the inference.
            model_class_names: The class names of the model.
            box_idxes: The index of the box in the output.
            boxes: The boxes of the output.
            node: (Class) The InferenceEngine object.
        Returns:
            None
        """
