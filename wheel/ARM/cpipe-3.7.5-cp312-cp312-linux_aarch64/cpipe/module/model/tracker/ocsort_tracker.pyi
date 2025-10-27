from _typeshed import Incomplete
from cpipe.module.model.tracker.ocsort_matching import associate as associate, iou_batch as iou_batch, linear_assignment as linear_assignment

def k_previous_obs(observations, cur_age, k): ...
def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
def speed_direction(bbox1, bbox2): ...

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.

    Args:
        bbox (np.array): bbox in [x1,y1,x2,y2,score] format.
        delta_t (int): delta_t of previous observation
    """
    count: int
    kf: Incomplete
    score: Incomplete
    time_since_update: int
    id: Incomplete
    history: Incomplete
    hits: int
    hit_streak: int
    age: int
    last_observation: Incomplete
    observations: Incomplete
    history_observations: Incomplete
    velocity: Incomplete
    delta_t: Incomplete
    def __init__(self, bbox, delta_t: int = 3) -> None: ...
    def update(self, bbox) -> None:
        """
        Updates the state vector with observed bbox.
        """
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
    def get_state(self): ...

class OCSORTTracker:
    """
    OCSORT tracker, support single class

    Args:
        det_thresh (float): threshold of detection score
        max_age (int): maximum number of missed misses before a track is deleted
        min_hits (int): minimum hits for associate
        iou_threshold (float): iou threshold for associate
        delta_t (int): delta_t of previous observation
        inertia (float): vdc_weight of angle_diff_cost for associate
        vertical_ratio (float): w/h, the vertical ratio of the bbox to filter
            bad results. If set <= 0 means no need to filter bboxes，usually set
            1.6 for pedestrian tracking.
        min_box_area (int): min box area to filter out low quality boxes
        use_byte (bool): Whether use ByteTracker, default False
    """
    det_thresh: Incomplete
    max_age: Incomplete
    min_hits: Incomplete
    iou_threshold: Incomplete
    delta_t: Incomplete
    inertia: Incomplete
    vertical_ratio: Incomplete
    min_box_area: Incomplete
    use_byte: Incomplete
    trackers: Incomplete
    frame_count: int
    def __init__(self, det_thresh: float = 0.6, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3, delta_t: int = 3, inertia: float = 0.2, vertical_ratio: int = -1, min_box_area: int = 0, use_byte: bool = False) -> None: ...
    def update(self, pred_dets, pred_embs=None):
        """
        Args:
            pred_dets (np.array): Detection results of the image, the shape is
                [N, 6], means 'cls_id, score, x0, y0, x1, y1'.
            pred_embs (np.array): Embedding results of the image, the shape is
                [N, 128] or [N, 512], default as None.

        Return:
            tracking boxes (np.array): [M, 6], means 'x0, y0, x1, y1, score, id'.
        """
