from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    # 本地USB 摄像头
    vs = VideoStreamer("streamers", 0, 3)

    # 视频流模式
    # vs = VideoStreamer("streamers", "rtmp://192.168.8.122:1935/live/7777", 3, 1)

    # 文件模式
    # vs = VideoStreamer("streamers","./test.mp4", queue_size, 1, once_mode=True)

    cp = CPipeInsight(http_insight=True)
    vs += cp

    Node.launch()
