import threading
from _typeshed import Incomplete

def clip_coords(boxes, img_shape) -> None:
    """
    Clip bounding xyxy bounding boxes to image shape (height
    Args:
        boxes: bounding boxes (x1, y1, x2, y2)
        img_shape: image shape (height, width)

    Returns: None

    """
def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
def non_max_suppression(prediction, conf_thres: float = 0.25, iou_thres: float = 0.45, classes=None, agnostic: bool = False, multi_label: bool = False, labels=(), max_det: int = 300, nc: int = 0, max_time_img: float = 0.05, max_nms: int = 30000, max_wh: int = 7680, in_place: bool = True, rotated: bool = False):
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
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    Args:
        img1_shape: original shape of the image
        coords: bounding boxes (x1, y1, x2, y2)
        img0_shape: target shape of the image
        ratio_pad: padding ratio

    Returns: rescaled bounding boxes

    """
def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto: bool = True, scaleFill: bool = False, scaleup: bool = True, stride: int = 32):
    """
    Resize image to a 32-pixel-multiple rectangle while keeping aspect ratio.
    """
def letterbox_cuda(img, new_shape=(640, 640), color=(114, 114, 114), auto: bool = True, scaleFill: bool = False, scaleup: bool = True, stride: int = 32, device: str = 'cuda:0'):
    """
    Resize image to a 32-pixel-multiple rectangle while keeping aspect ratio.
    """
def preprocess_yolov7(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for YOLOv7.
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """
def preprocess_yolov7_rknn(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for YOLOv7.
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """
def preprocess_yolov7_cuda(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for YOLOv7.
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """
def preprocess_retinaface(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for RetinaFace.
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """
def embedding_preprocess(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for embedding model.
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """
def class_preprocess(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for classification
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """
def mm_class_preprocess(self, raw_bgr_image, num) -> None:
    """
    Preprocess the raw image for classification
    Args:
        self: ProcessThread object
        raw_bgr_image: input image
        num: image index

    Returns: None

    """
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
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
def scale_masks(masks, shape, padding: bool = True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        masks (torch.Tensor): The returned masks with dimensions [h, w, n]
    """

class ProcessThread(threading.Thread):
    images: Incomplete
    out_img: Incomplete
    input_w: Incomplete
    input_h: Incomplete
    idx: Incomplete
    worker_num: Incomplete
    device: Incomplete
    batch_scale: Incomplete
    mean: Incomplete
    std: Incomplete
    auto: bool
    half: bool
    area_flag: Incomplete
    area_info: Incomplete
    area_info_streamer_names: Incomplete
    args: Incomplete
    kwargs: Incomplete
    preprocessor: Incomplete
    def __init__(self, idx, worker_num, images, out_img, device, input_w: int = 640, input_h: int = 640, preprocessor=None, batch_scale=None, area_flag: bool = False, area_info=None, area_info_streamer_names=None, mean=None, std=None, *args, **kwargs) -> None:
        """
        CPipe process thread class. Used for multi-thread task processing.
        Args:
            idx: start index
            worker_num: number of workers
            images: input images
            out_img: output images
            device: device of the model
            input_w: input width
            input_h: input height
            preprocessor: preprocessor function
            batch_scale: batch scale data
            area_flag: area flag for area mask
            area_info: area information
            area_info_streamer_names: area information streamer names
            mean: process mean
            std: process std
            *args:
            **kwargs:
        """
    def run(self) -> None:
        """
        Run the process thread.
        Returns: None

        """

def load_data(input_images, device, input_size_hw, num_workers, preprocessor=None, area_flag: bool = False, area_info=None, area_info_streamer_names=None, mean=None, std=None, batch_scale: bool = False, *args, **kwargs):
    """
    Entry function of data preprocessing method for all models in CPipe.
    Args:
        input_images: input images
        device: device of the model
        input_size_hw: input size
        num_workers: number of workers
        preprocessor: preprocessor function
        area_flag: area flag for area mask
        area_info: area information
        area_info_streamer_names: area information streamer names
        mean: process mean
        std: process std
        batch_scale: batch scale flag
        *args:
        **kwargs:
    """
