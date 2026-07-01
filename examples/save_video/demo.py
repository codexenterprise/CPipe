from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":
    # video stream mode
    streamer1 = VideoStreamer(node_name="streamers", stream="/mnt/d/2026-02-24_RT16000282878016CN_23_104700_RI304.mp4", process_frame_interval=1, once_mode=True)

    # save video when the streamer node is finished
    # streamer1 = VideoStreamer(node_name="streamers", stream="./test.mp4", process_frame_interval=1, once_mode=True)

    chache = YOLOv10(node_name="chache",
                    model_path="model_files/yolov10n.engine",
                    # input_size=(3, 640, 640),
                    valid_class_names=["person"],
                    # max_batch_size=1,
                    # save_top_n_objects=32,
                    # area_flag=True
                    )

    cpipeinsight = CPipeInsight(http_insight=True, save_video=True)  # save_video must be True

    streamer1 += [chache, cpipeinsight]

    Node.launch(check_node=True, check_interval=5, auto_restart=False)  # auto_restart must be False
