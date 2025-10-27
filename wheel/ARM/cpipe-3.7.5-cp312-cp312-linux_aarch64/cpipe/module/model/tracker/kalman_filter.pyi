from _typeshed import Incomplete

chi2inv95: Incomplete

class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """
    _motion_mat: Incomplete
    _update_mat: Incomplete
    _std_weight_position: Incomplete
    _std_weight_velocity: Incomplete
    def __init__(self) -> None: ...
    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with
                center position (x, y), aspect ratio a, and height h.

        Returns:
            The mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are 
            initialized to 0 mean.
        """
    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state
                at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the
                object state at the previous time step.

        Returns:
            The mean vector and covariance matrix of the predicted state. 
            Unobserved velocities are initialized to 0 mean.
        """
    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.

        Args
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            The projected mean and covariance matrix of the given state estimate.
        """
    def multi_predict(self, mean, covariance):
        """
        Run Kalman filter prediction step (Vectorized version).
        
        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states
                at the previous time step.
            covariance (ndarray): The Nx8x8 dimensional covariance matrics of the
                object states at the previous time step.

        Returns:
            The mean vector and covariance matrix of the predicted state.
            Unobserved velocities are initialized to 0 mean.
        """
    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector
                (x, y, a, h), where (x, y) is the center position, a the aspect
                ratio, and h the height of the bounding box.

        Returns:
            The measurement-corrected state distribution.
        """
    def gating_distance(self, mean, covariance, measurements, only_position: bool = False, metric: str = 'maha'):
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        
        Args:
            mean (ndarray): Mean vector over the state distribution (8
                dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8
                dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N measurements,
                each in format (x, y, a, h) where (x, y) is the bounding box center
                position, a the aspect ratio, and h the height.
            only_position (Optional[bool]): If True, distance computation is 
                done with respect to the bounding box center position only.
            metric (str): Metric type, 'gaussian' or 'maha'.

        Returns
            An array of length N, where the i-th element contains the squared
            Mahalanobis distance between (mean, covariance) and `measurements[i]`.
        """
