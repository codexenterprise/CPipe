from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    # local USB camera
    vs = VideoStreamer(node_name="streamers", stream=0, queue_size=3) 

    # video stream mode
    # vs = VideoStreamer(node_name="streamers", stream="rtmp://192.168.8.122:1935/live/7777", process_frame_interval=1, queue_size=3) 

    # file mode
    # vs = VideoStreamer(node_name="streamers", stream="./test.mp4", process_frame_interval=1, once_mode=True, queue_size=3) 

    cp = CPipeInsight(http_insight=True)
    vs += cp

    Node.launch()
