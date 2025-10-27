from _typeshed import Incomplete
from cpipe.module.cinferencer import CDetector as CDetector
from cpipe.module.cmodel import Cmodel as Cmodel
from cpipe.module.dataprocessing import preprocess_yolov7 as preprocess_yolov7, preprocess_yolov7_rknn as preprocess_yolov7_rknn, scale_coords as scale_coords

class YOLOv7(CDetector):
    preprocessor: Incomplete
    anchor: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, valid_class_names=None, max_batch_size: int = 1, conf_thres: float = 0.25, iou_thres: float = 0.45, anchor=None, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, save_top_n_objects=None, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        """
        YOLOv7 is a class for YOLOv7 model.

        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 416, 416]
            class_names: (list) The class names.
            valid_class_names: (list) The valid class names.
            max_batch_size: (int) The max batch size.
            conf_thres: (float) The confidence threshold.
            iou_thres: (float) The iou threshold.
            anchor: (list) The anchor. e.g. np.array([12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0]).reshape(3, -1, 2).tolist()
            warmup: (bool) The warmup flag.
            device: (str) The device.
            threading_num: (int) The threading number.
            save_top_n_objects: (int) The save top n objects.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input names.
            output_names: (list) The output names.
            gray_mode: (bool) Whether to use gray mode.

        Returns: None

        """
    def infer(self, inputs, *args, **kwargs):
        """
        The infer function of the model
        Args:
            inputs: inputs[0] is pre_imgs, inputs[1] is origin_imgs
            *args:  frames_stream_names. eg. ['stream1', 'stream2']
            **kwargs:

        Returns:

        """
    def box_process(self, position, anchors): ...
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
    def post_process(self, input_data, anchors): ...
    def infer_onnx(self, inputs, *args, **kwargs):
        """
        The infer function of the model
        Args:
            inputs: inputs[0] is pre_imgs, inputs[1] is origin_imgs
            *args:  frames_stream_names. eg. ['stream1', 'stream2']
            **kwargs:

        Returns:

        """
