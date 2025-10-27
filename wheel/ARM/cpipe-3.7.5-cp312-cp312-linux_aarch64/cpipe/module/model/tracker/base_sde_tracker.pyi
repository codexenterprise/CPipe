from _typeshed import Incomplete

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """
    Tentative: int
    Confirmed: int
    Deleted: int

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Args:
        mean (ndarray): Mean vector of the initial state distribution.
        covariance (ndarray): Covariance matrix of the initial state distribution.
        track_id (int): A unique track identifier.
        n_init (int): Number of consecutive detections before the track is confirmed.
            The track state is set to `Deleted` if a miss occurs within the first
            `n_init` frames.
        max_age (int): The maximum number of consecutive misses before the track
            state is set to `Deleted`.
        cls_id (int): The category id of the tracked box.
        score (float): The confidence score of the tracked box.
        feature (Optional[ndarray]): Feature vector of the detection this track
            originates from. If not None, this feature is added to the `features` cache.

    Attributes:
        hits (int): Total number of measurement updates.
        age (int): Total number of frames since first occurance.
        time_since_update (int): Total number of frames since last measurement
            update.
        state (TrackState): The current track state.
        features (List[ndarray]): A cache of features. On each measurement update,
            the associated feature vector is added to this list.
    """
    mean: Incomplete
    covariance: Incomplete
    track_id: Incomplete
    hits: int
    age: int
    time_since_update: int
    cls_id: Incomplete
    score: Incomplete
    start_time: Incomplete
    state: Incomplete
    features: Incomplete
    feat: Incomplete
    _n_init: Incomplete
    _max_age: Incomplete
    last_observation: Incomplete
    history_observations: Incomplete
    def __init__(self, mean, covariance, track_id, n_init, max_age, cls_id, score, feature=None) -> None: ...
    def to_tlwh(self):
        """Get position in format `(top left x, top left y, width, height)`."""
    def to_tlbr(self):
        """Get position in bounding box format `(min x, miny, max x, max y)`."""
    def predict(self, kalman_filter) -> None:
        """
        Propagate the state distribution to the current time step using a Kalman
        filter prediction step.
        """
    def update(self, kalman_filter, detection) -> None:
        """
        Perform Kalman filter measurement update step and update the associated
        detection feature cache.
        """
    def mark_missed(self) -> None:
        """Mark this track as missed (no association at the current time step).
        """
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
    def is_confirmed(self):
        """Returns True if this track is confirmed."""
    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
