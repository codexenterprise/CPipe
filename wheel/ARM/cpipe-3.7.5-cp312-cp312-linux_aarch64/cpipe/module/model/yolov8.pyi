import numpy as np
import torch
from _typeshed import Incomplete
from cpipe.module.cinferencer import COBBDetector as COBBDetector
from cpipe.module.dataprocessing import clip_coords as clip_coords, preprocess_yolov7 as preprocess_yolov7

def prob_iou_gpu(obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-07) -> torch.Tensor:
    """
    Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obb_bbox, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (M, 5) representing predicted obb_bbox, with xywhr format.
        eps (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: A tensor of shape (N, M) representing obb similarities.
    """
def prob_iou_cpu(obb1, obb2, CIoU: bool = False, eps: float = 1e-07) -> np.ndarray:
    """
    Calculate the prob iou between oriented bounding boxes using numpy on CPU.

    Args:
        obb1 (np.ndarray): A numpy array of shape (N, 5) representing ground truth obb_bbox, with xywhr format.
        obb2 (np.ndarray): A numpy array of shape (M, 5) representing predicted obb_bbox, with xywhr format.
        eps (float): A small value to avoid division by zero.

    Returns:
        np.ndarray: A numpy array of shape (N, M) representing obb similarities.
    """
def _get_covariance_matrix(boxes):
    """
    Generate covariance matrix from obbs.

    Args:
        boxes (np.ndarray or torch.Tensor): A numpy array of shape (N, 5) or a torch tensor with xywhr format.

    Returns:
        (tuple): Covariance matrices corresponding to original rotated bounding boxes.
    """
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None): ...
def xywhr2xyxyxyxy_with_bbox(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert bounding boxes from xywhr format to four corner points and enclosing rectangle coordinates.

    Args:
        boxes (np.ndarray): Bounding boxes in xywhr format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - np.ndarray: Bounding boxes in xyxyxyxy format (four corner points).
            - np.ndarray: Bounding boxes in enclosing rectangle format (xmin, ymin, xmax, ymax).
    """

class YOLOv8obb(COBBDetector):
    origin_images: Incomplete
    preprocessor: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, valid_class_names=None, max_batch_size: int = 1, conf_thres: float = 0.25, iou_thres: float = 0.45, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, save_top_n_objects=None, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        """
        YOLOv8obb is a class for YOLOv8 model.

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
            *args:  frames_stream_names. e.g. ['stream1', 'stream2']
            **kwargs:

        Returns:

        """
    def postprocess(self, inputs, *args, **kwargs):
        """
        Post-process the prediction results to filter and format the bounding boxes using numpy on CPU.

        Args:
            data (np.ndarray): The prediction output from the model. It is expected to be a numpy array of shape (
            batch_size, num_boxes, class_scores+5), where each box has class_scores+5 values (x, y, w, h, class_scores[
            x], angle).
            **kwargs: Additional keyword arguments.

        Returns:
            List[np.ndarray]: A list of numpy arrays, where each array contains the processed bounding boxes for
            each image in the batch. Each bounding box is represented as an array of shape (14,), containing: [x1, y1,
            x2, y2, x3, y3, x4, y4, xmin, ymin, xmax, ymax, confidence, class]. Here:
            - (x1, y1), (x2, y2), (x3, y3), (x4, y4) are the four corner points of the rotated bounding box.
            - (xmin, ymin) and (xmax, ymax) are the coordinates of the enclosing rectangle.
        """
    @staticmethod
    def obb_nms_cpu(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Non-Maximum Suppression (NMS) for oriented bounding boxes using numpy on CPU.

        Args:
            boxes (np.ndarray): Oriented bounding boxes in xywhr format.
            scores (np.ndarray): Scores for each box.
            iou_threshold (float): IOU threshold for NMS.

        Returns:
            np.ndarray: Indices of the kept boxes.
        """
