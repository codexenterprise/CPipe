from .base_sde_tracker import Track as Track
from .deepsort_matching import NearestNeighborDistanceMetric as NearestNeighborDistanceMetric, gate_cost_matrix as gate_cost_matrix, iou_cost as iou_cost, matching_cascade as matching_cascade, min_cost_matching as min_cost_matching
from .kalman_filter import KalmanFilter as KalmanFilter
from _typeshed import Incomplete

class Detection:
    """
    This class represents a bounding box detection in a single image.

    Args:
        tlwh (Tensor): Bounding box in format `(top left x, top left y,
            width, height)`.
        score (Tensor): Bounding box confidence score.
        feature (Tensor): A feature vector that describes the object
            contained in this image.
        cls_id (Tensor): Bounding box category id.
    """
    tlwh: Incomplete
    score: Incomplete
    feature: Incomplete
    cls_id: Incomplete
    def __init__(self, tlwh, score, feature, cls_id) -> None: ...
    def to_tlbr(self):
        """
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
    def to_xyah(self):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """

class DeepSORTTracker:
    '''
    DeepSORT tracker

    Args:
        input_size (list): input feature map size to reid model, [h, w] format,
            [64, 192] as default.
        min_box_area (int): min box area to filter out low quality boxes
        vertical_ratio (float): w/h, the vertical ratio of the bbox to filter
            bad results, set 1.6 default for pedestrian tracking. If set <=0
            means no need to filter bboxes.
        budget (int): If not None, fix samples per class to at most this number.
            Removes the oldest samples when the budget is reached.
        max_age (int): maximum number of missed misses before a track is deleted
        n_init (float): Number of frames that a track remains in initialization
            phase. Number of consecutive detections before the track is confirmed. 
            The track state is set to `Deleted` if a miss occurs within the first 
            `n_init` frames.
        metric_type (str): either "euclidean" or "cosine", the distance metric 
            used for measurement to track association.
        matching_threshold (float): samples with larger distance are 
            considered an invalid match.
        max_iou_distance (float): max iou distance threshold
        motion (object): KalmanFilter instance
    '''
    input_size: Incomplete
    min_box_area: Incomplete
    vertical_ratio: Incomplete
    max_age: Incomplete
    n_init: Incomplete
    metric: Incomplete
    max_iou_distance: Incomplete
    motion: Incomplete
    tracks: Incomplete
    _next_id: int
    def __init__(self, input_size=[64, 192], min_box_area: int = 0, vertical_ratio: int = -1, budget: int = 100, max_age: int = 70, n_init: int = 3, metric_type: str = 'cosine', matching_threshold: float = 0.2, max_iou_distance: float = 0.9, motion: str = 'KalmanFilter') -> None: ...
    def predict(self) -> None:
        """
        Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
    def update(self, pred_dets, pred_embs):
        """
        Perform measurement update and track management.
        Args:
            pred_dets (np.array): Detection results of the image, the shape is
                [N, 6], means 'cls_id, score, x0, y0, x1, y1'.
            pred_embs (np.array): Embedding results of the image, the shape is
                [N, 128], usually pred_embs.shape[1] is a multiple of 128.
        """
    def _match(self, detections): ...
    def _initiate_track(self, detection) -> None: ...
