from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":
    stream1 = VideoStreamer(stream="/mnt/d/坑洼.MP4", process_frame_interval=2, once_mode=True)
    detect = YOLOv7(
                    model_path="/mnt/d/kengwa_batch1.engine",
                    class_names=['Manhole cover', 'hollow'],
                    save_top_n_objects=32,
                    area_flag=False
                    )

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=1, save_video=True, chinese_font_size=30)

    stream1 += [detect, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False, agent=False)
