from cpipe.module.model.tracker.tracker import Tracker
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":
    # 视频流模式
    streamer1 = VideoStreamer("streamers", "rtmp://192.168.8.122:1935/live/7777", 3, 1)

    # 文件模式, once_mode必须为True
    # streamer1 = VideoStreamer("streamers","./test.mp4", queue_size, 1, once_mode=True)

    chache = YOLOv7("chache",
                    "../../src/dongsheng/dongsheng_huowu_new.engine",
                    3,
                    inputSize=(3, 640, 640),
                    class_names=['materials'],
                    max_batch_size=1,
                    conf_thres=0.4,
                    )

    cpipeinsight = CPipeInsight(http_insight=True, save_video=True)  # save_video必须为True

    streamer1 += [chache, cpipeinsight]

    Node.launch(check_node=True, check_interval=5, auto_restart=False)  # auto_restart必须为False
