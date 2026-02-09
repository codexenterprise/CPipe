from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":
    # video stream mode
    streamer1 = VideoStreamer(node_name="streamers", stream="rtmp://192.168.8.122:1935/live/7777", process_frame_interval=1, once_mode=True)

    # save video when the streamer node is finished
    # streamer1 = VideoStreamer(node_name="streamers", stream="./test.mp4", process_frame_interval=1, once_mode=True)

    chache = YOLOv7(node_name="chache",
                    model_path="../../src/dongsheng/dongsheng_huowu_new.engine",
                    input_size=(3, 640, 640),
                    class_names=['materials'],
                    max_batch_size=1,
                    conf_thres=0.4
                    )

    cpipeinsight = CPipeInsight(http_insight=True, save_video=True)  # save_video must be True

    streamer1 += [chache, cpipeinsight]

    Node.launch(check_node=True, check_interval=5, auto_restart=False)  # auto_restart must be False
