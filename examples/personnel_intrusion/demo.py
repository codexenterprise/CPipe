from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from examples.personnel_intrusion.custom_node import PersonnelIntrusion

if __name__ == "__main__":
    stream1 = VideoStreamer(stream="rtmp://192.168.10.7:1935/live/7777", process_frame_interval=1, once_mode=True)
    detect = YOLOv10(
                    model_path="model_files/yolov10n_batch1.engine",
                    # class_names=['person', ...],
                    valid_class_names=["person"],
                    save_top_n_objects=32,
                    area_flag=True
                    )

    cpipeinsight = CPipeInsight(http_insight=True)

    personnel_intrusion = PersonnelIntrusion()

    stream1 += [detect, personnel_intrusion, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False)
