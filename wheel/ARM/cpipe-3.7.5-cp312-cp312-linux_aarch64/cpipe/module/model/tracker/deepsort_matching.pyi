from . import kalman_filter as kalman_filter
from _typeshed import Incomplete

INFTY_COST: float

def iou_1toN(bbox, candidates):
    """
    Computer intersection over union (IoU) by one box to N candidates.

    Args:
        bbox (ndarray): A bounding box in format `(top left x, top left y, width, height)`.
            candidates (ndarray): A matrix of candidate bounding boxes (one per row) in the
            same format as `bbox`.

    Returns:
        ious (ndarray): The intersection over union in [0, 1] between the `bbox`
            and each candidate. A higher score means a larger fraction of the
            `bbox` is occluded by the candidate.
    """
def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    IoU distance metric.

    Args:
        tracks (list[Track]): A list of tracks.
        detections (list[Detection]): A list of detections.
        track_indices (Optional[list[int]]): A list of indices to tracks that
            should be matched. Defaults to all `tracks`.
        detection_indices (Optional[list[int]]): A list of indices to detections
            that should be matched. Defaults to all `detections`.

    Returns:
        cost_matrix (ndarray): A cost matrix of shape len(track_indices), 
            len(detection_indices) where entry (i, j) is 
            `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
    """
def _nn_euclidean_distance(s, q):
    """
    Compute pair-wise squared (Euclidean) distance between points in `s` and `q`.

    Args:
        s (ndarray): Sample points: an NxM matrix of N samples of dimensionality M.
        q (ndarray): Query points: an LxM matrix of L samples of dimensionality M.

    Returns:
        distances (ndarray): A vector of length M that contains for each entry in `q` the
            smallest Euclidean distance to a sample in `s`.
    """
def _nn_cosine_distance(s, q):
    """
    Compute pair-wise cosine distance between points in `s` and `q`.

    Args:
        s (ndarray): Sample points: an NxM matrix of N samples of dimensionality M.
        q (ndarray): Query points: an LxM matrix of L samples of dimensionality M.

    Returns:
        distances (ndarray): A vector of length M that contains for each entry in `q` the
            smallest Euclidean distance to a sample in `s`.
    """

class NearestNeighborDistanceMetric:
    '''
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Args:
        metric (str): Either "euclidean" or "cosine".
        matching_threshold (float): The matching threshold. Samples with larger
            distance are considered an invalid match.
        budget (Optional[int]): If not None, fix samples per class to at most
            this number. Removes the oldest samples when the budget is reached.

    Attributes: 
        samples (Dict[int -> List[ndarray]]): A dictionary that maps from target
            identities to the list of samples that have been observed so far.
    '''
    _metric: Incomplete
    matching_threshold: Incomplete
    budget: Incomplete
    samples: Incomplete
    def __init__(self, metric, matching_threshold, budget=None) -> None: ...
    def partial_fit(self, features, targets, active_targets) -> None:
        """
        Update the distance metric with new data.

        Args:
            features (ndarray): An NxM matrix of N features of dimensionality M.
            targets (ndarray): An integer array of associated target identities.
            active_targets (List[int]): A list of targets that are currently
                present in the scene.
        """
    def distance(self, features, targets):
        """
        Compute distance between features and targets.

        Args:
            features (ndarray): An NxM matrix of N features of dimensionality M.
            targets (list[int]): A list of targets to match the given `features` against.

        Returns:
            cost_matrix (ndarray): a cost matrix of shape len(targets), len(features),
                where element (i, j) contains the closest squared distance between
                `targets[i]` and `features[j]`.
        """

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    """
    Solve linear assignment problem.

    Args:
        distance_metric :
            Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
            The distance metric is given a list of tracks and detections as 
            well as a list of N track indices and M detection indices. The 
            metric should return the NxM dimensional cost matrix, where element
            (i, j) is the association cost between the i-th track in the given
            track indices and the j-th detection in the given detection_indices.
        max_distance (float): Gating threshold. Associations with cost larger
            than this value are disregarded.
        tracks (list[Track]): A list of predicted tracks at the current time
            step.
        detections (list[Detection]): A list of detections at the current time
            step.
        track_indices (list[int]): List of track indices that maps rows in
            `cost_matrix` to tracks in `tracks`.
        detection_indices (List[int]): List of detection indices that maps
            columns in `cost_matrix` to detections in `detections`.

    Returns:
        A tuple (List[(int, int)], List[int], List[int]) with the following
        three entries:
            * A list of matched track and detection indices.
            * A list of unmatched track indices.
            * A list of unmatched detection indices.
    """
def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections, track_indices=None, detection_indices=None):
    """
    Run matching cascade.

    Args:
        distance_metric :
            Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
            The distance metric is given a list of tracks and detections as 
            well as a list of N track indices and M detection indices. The 
            metric should return the NxM dimensional cost matrix, where element
            (i, j) is the association cost between the i-th track in the given
            track indices and the j-th detection in the given detection_indices.
        max_distance (float): Gating threshold. Associations with cost larger
            than this value are disregarded.
        cascade_depth (int): The cascade depth, should be se to the maximum
            track age.
        tracks (list[Track]): A list of predicted tracks at the current time
            step.
        detections (list[Detection]): A list of detections at the current time
            step.
        track_indices (list[int]): List of track indices that maps rows in
            `cost_matrix` to tracks in `tracks`.
        detection_indices (List[int]): List of detection indices that maps
            columns in `cost_matrix` to detections in `detections`.

    Returns:
        A tuple (List[(int, int)], List[int], List[int]) with the following
        three entries:
            * A list of matched track and detection indices.
            * A list of unmatched track indices.
            * A list of unmatched detection indices.
    """
def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices, gated_cost=..., only_position: bool = False):
    """
    Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Args:
        kf (object): The Kalman filter.
        cost_matrix (ndarray): The NxM dimensional cost matrix, where N is the
            number of track indices and M is the number of detection indices,
            such that entry (i, j) is the association cost between
            `tracks[track_indices[i]]` and `detections[detection_indices[j]]`.
        tracks (list[Track]): A list of predicted tracks at the current time
            step.
        detections (list[Detection]): A list of detections at the current time
            step.
        track_indices (List[int]): List of track indices that maps rows in
            `cost_matrix` to tracks in `tracks`.
        detection_indices (List[int]): List of detection indices that maps
            columns in `cost_matrix` to detections in `detections`.
        gated_cost (Optional[float]): Entries in the cost matrix corresponding
            to infeasible associations are set this value. Defaults to a very
            large value.
        only_position (Optional[bool]): If True, only the x, y position of the
            state distribution is considered during gating. Default False.
    """
