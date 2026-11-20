import numpy as np

from cpipe.module.insight import CPipeInsight
from cpipe.module.model.tracker.tracker import Tracker
from cpipe.module.model.yolo26 import YOLO26
from cpipe.module.node import Node
from cpipe.module.reports.alarmreport import AlarmReport
from cpipe.module.streamer import VideoStreamer
from my_logic import my_logic

if __name__ == "__main__":
    stream1 = VideoStreamer(stream="rtsp://admin:codexwell!@192.168.10.222:554/Streaming/Channels/101", process_frame_interval=5)
    detect = YOLO26(model_path="model_files/yolo26n_person.engine")
    # video alarm needs track_id, so the tracker is required
    tracker = Tracker(tracker_type="ocsort", secondary_class_names=["person"])
    logic = my_logic(vote_x=3, vote_y=5)
    alarm = AlarmReport(store_dir="./alarm_data", video_seconds=3, retention_days=7,
                        mqtt_broker="127.0.0.1",
                        mqtt_port=1883,
                        mqtt_http_base="http://127.0.0.1:9966"
     )
    cpipeinsight = CPipeInsight(http_insight=True)
    stream1 += detect
    detect += [tracker, logic, alarm, cpipeinsight]
    
    Node.launch(check_node=True, auto_restart=True)
