from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node



if __name__ == "__main__":
    # args:
    #         show_polygon_box: self.kwargs.get("show_polygon_box", False)
    #         show_box: self.kwargs.get("show_box", True)
    #         show_box_name: self.kwargs.get("show_box_name", True)
    #         show_polygon: self.kwargs.get("show_polygon", True)
    #         show_mask: self.kwargs.get("show_mask", True)
    #         show_key_points: self.kwargs.get("show_key_points", True)
    #         show_person: self.kwargs.get("show_person", True)
    #         show_classification: self.kwargs.get("show_classification", True)
    #         show_track: self.kwargs.get("show_track", True)

    cpipeinsight = CPipeInsight(http_insight=True,
                                show_box=False, # not show box
                                show_box_name=False, # not show box name
                                show_polygon=False # not show polygon
                                # ...
                                )

