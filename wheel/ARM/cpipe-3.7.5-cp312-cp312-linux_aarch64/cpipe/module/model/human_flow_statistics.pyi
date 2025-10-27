from _typeshed import Incomplete
from cpipe.config.config import CLOGER_LEVEL as CLOGER_LEVEL, CLOGER_LEVEL_DEBUG as CLOGER_LEVEL_DEBUG
from cpipe.module.inferenceengine import InferenceEngine as InferenceEngine
from cpipe.module.node import Node as Node

class HumanFLowStatistics(InferenceEngine):
    in_and_out_line: Incomplete
    in_and_out_count: Incomplete
    list_overlapping_line_in: Incomplete
    list_overlapping_line_out: Incomplete
    in_out_masks: Incomplete
    missing_count: Incomplete
    missing_count_thres: Incomplete
    dump_images: Incomplete
    def __init__(self, nodeName: str, queue_size: int, miss_count_thres: int = 3, area_flag: bool = True, secondary_class_names=None, dump_images: bool = True, *args, **kwargs) -> None:
        """
        HumanFLowStatistics is a class for HumanFLowStatistics
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The queue size.
            miss_count_thres: (int) The missing count threshold.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            dump_images: (bool) The dump images flag.
            *args:
            **kwargs:
        """
    def get_streamer_area(self) -> None: ...
    def _start(self) -> None: ...
    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs) -> None: ...
    def to_cdata_secondary(self, pred, new_cdata, streamer_names, box_idxes, boxes, *args, **kwargs) -> None: ...
    def is_cross_line(self, center_x: int, center_y: int, track_id: int, name: str):
        """
        judge whether the track is across the line.
        Args:
            center_x: (int) The center x.
            center_y: (int) The center y.
            track_id: (int) The track id.
            name: (str) The name.

        """
