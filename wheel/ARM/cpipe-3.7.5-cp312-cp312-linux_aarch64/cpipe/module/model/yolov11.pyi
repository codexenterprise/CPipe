from _typeshed import Incomplete
from cpipe.module.cinferencer import CInstanceSegmentation as CInstanceSegmentation
from cpipe.module.cmodel import Cmodel as Cmodel
from cpipe.module.dataprocessing import preprocess_yolov7 as preprocess_yolov7
from cpipe.module.model.yolov7 import YOLOv7 as YOLOv7

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
def non_max_suppression(prediction, conf_thres: float = 0.25, iou_thres: float = 0.45, classes=None, agnostic: bool = False, multi_label: bool = False, labels=(), max_det: int = 300, nc: int = 0, max_time_img: float = 0.05, max_nms: int = 30000, max_wh: int = 7680):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

class YOLOv11InstanceSeg(CInstanceSegmentation):
    input_h: Incomplete
    input_w: Incomplete
    preprocessor: Incomplete
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, valid_class_names=None, max_batch_size: int = 1, conf_thres: float = 0.25, iou_thres: float = 0.45, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, save_top_n_objects=None, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, return_masks: bool = False, *args, **kwargs) -> None:
        """
        YOLOv11Seg is a class for YOLOv11 Segment model.

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
            return_masks: (bool) Whether to return mask.

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

class YOLOv11(YOLOv7):
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, valid_class_names=None, max_batch_size: int = 1, conf_thres: float = 0.25, iou_thres: float = 0.45, warmup: bool = True, device: str = 'cuda:0', threading_num: int = 4, save_top_n_objects=None, area_flag: bool = False, secondary_class_names=None, input_names=None, output_names=None, gray_mode: bool = False, *args, **kwargs) -> None:
        """
        YOLOv11TRT is a class for YOLOv11 TensorRT model.

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
