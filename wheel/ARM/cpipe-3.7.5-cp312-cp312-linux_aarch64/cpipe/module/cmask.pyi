from _typeshed import Incomplete
from cpipe.config.config import CMASK_YAML_PATH as CMASK_YAML_PATH, SHARE_MEMORY_MAX_HEIGHT_WIDTH_CHANNEL as SHARE_MEMORY_MAX_HEIGHT_WIDTH_CHANNEL
from cpipe.module.clogger import CLogger as CLogger
from cpipe.module.utils import CConfig as CConfig

class Mask:
    MASK_TYPE_POLYGON: str
    MASK_TYPE_LINE: str
    mask_type: Incomplete
    mask_name: Incomplete
    mask_coords: Incomplete
    mask_classes: Incomplete
    mask_confs: Incomplete
    mask_map_init: bool
    coords_init: bool
    mask_map_shared_memory: Incomplete
    mask_map_shared_memory_shape: Incomplete
    mask_coords_with_wh_shared_memory: Incomplete
    mask_coords_with_wh_shared_memory_shape: Incomplete
    mask_roi: Incomplete
    _mask_map: Incomplete
    _mask_coords_with_wh: Incomplete
    def __init__(self, mask_name, mask_type, mask_coords=None, mask_classes=None, mask_confs=None) -> None:
        '''
        CPipe base class for storing masks.
        Args:
            mask_name: The name of the mask.
            mask_type: The type of the mask, "polygon" or "line".
            mask_coords: The coordinates of the mask. [[x1, y1], [x2, y2],...]
            mask_classes: The classes of the mask.
            mask_confs: The confidence of the mask.
        '''
    @property
    def mask_coords_with_wh(self): ...
    @property
    def mask_map(self): ...
    def check(self):
        """
        Check the mask.
        Returns: (bool, str) Whether the mask is valid and the error message.

        """
    def get_mask_map(self, streamer_hw, fill_num: int = 255, need_update_mask_map: bool = False):
        """
        Get the mask map.
        Args:
            streamer_hw: The height and width of the streamer.
            fill_num: The fill number.
            need_update_mask_map: Whether to update the mask map.
        Returns: (list, np.ndarray, list) The region of interest, the mask map, and the class and confidence.

        """
    def get_coords_with_wh(self, streamer_hw, need_update_mask_map: bool = False):
        """
        Get the coordinates with width and height.
        Args:
            streamer_hw: The height and width of the streamer.
            need_update_mask_map: Whether to update the mask map.

        Returns: The coordinates with width and height.

        """
    def update_mask_map(self, streamer_hw) -> None:
        """
        Update the mask map.
        Args:
            streamer_hw: The height and width of the streamer.
        """
    def get_info(self):
        """
        Get the information of the mask.
        Returns: (str) The information of the mask.
        """

class CMask:
    __allCMask__: Incomplete
    logger: Incomplete
    name: Incomplete
    _streamer_hw: Incomplete
    streamer_hw_bak: Incomplete
    sport_type: Incomplete
    masks: Incomplete
    def __init__(self, name, streamer_hw, sport_type=...) -> None:
        '''
        Each Streamer in CPipe corresponds to a CMask, which is used to store Mask information. Each Mask corresponds to the mask information of a node.
        Args:
            name: The name of the CMask.
            streamer_hw: The height and width of the streamer. [height, width], height and width are multiprocessing.Value.
            sport_type: The type of the mask, "polygon" or "line".
        '''
    def get_node_mask_info(self, node_name, mask_name=None):
        """
        Get one node mask information.
        Args:
            node_name: The name of the node.
            mask_name: The name of the mask.
        Returns: (str) The information of the mask.
        """
    def update_mask_map(self) -> None:
        """
        Update the mask map.
        """
    @property
    def streamer_hw(self): ...
    def check(self):
        """
        Check the mask.
        Returns: (bool, str) Whether the mask is valid and the error message.

        """
    def get_lines(self, node_name):
        """
        Get the lines.
        Args:
            node_name: The name of the node.

        Returns: The lines of the node.

        """
    @classmethod
    def find_mask(cls, mask_name, node_name=None, streamer_name=None):
        """
        Find the mask.
        Args:
            mask_name: The name of the mask.
            node_name: The name of the node.
            streamer_name: The name of the streamer.
        """
    def get_mask_maps(self, node_name, mask_fill_num_dict=None, mask_with_full_figure: bool = False):
        """
        Get the mask maps.
        Args:
            node_name: The name of the node.
            mask_fill_num_dict: The fill number of the mask.
            mask_with_full_figure: Whether to use the full figure.

        Returns: (list, np.ndarray, list) The region of interest, the mask map, and the class and confidence.

        """
    def get_mask_rois(self, node_name=None):
        """
        Get the mask rois.
        Args:
            node_name: The name of the node.

        Returns: (list, list) The mask rois and the mask names.

        """
    def add_polygon(self, node_name, mask_name, polygons=None, classes=None, confs=None):
        """
        Add the polygon.
        Args:
            node_name: The name of the node.
            mask_name: The name of the mask.
            polygons: The polygons. [[[x1, y1], [x2, y2],...], [[x1, y1], [x2, y2],...],...]
            classes: The classes. [class1, class2,...]
            confs: The confidences. [conf1, conf2,...]

        Returns: (bool, str) Whether the mask is valid and the error message.

        """
    def add_line(self, node_name, mask_name, lines=None):
        """
        Add the line.
        Args:
            node_name: The name of the node.
            mask_name: The name of the mask.
            lines: The lines. [[[x1, y1], [x2, y2]], [[x1, y1], [x2, y2]],...]

        Returns: (bool, str) Whether the mask is valid and the error message.

        """
    def del_mask(self, node_name, mask_name):
        """
        Delete the mask.
        Args:
            node_name: Delete the mask of the node.
            mask_name: The name of the mask.

        Returns: (bool, str) Whether the mask is valid and the error message.

        """
    def to_json(self):
        """
        Convert the mask to json.
        Returns: The json of the CMask.

        """
    @classmethod
    def save_yaml(cls, node) -> None:
        """
        Save the yaml.
        Args:
            node: The node class. To send event to all nodes.

        Returns: None

        """
    @classmethod
    def load_yaml(cls, show_log: bool = True) -> None:
        """
        Load the yaml.
        Returns: None

        """
