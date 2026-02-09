
import time
from cpipe.module.node import Node
import threading
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight

if __name__ == "__main__":
    streamer = VideoStreamer(
        node_name="VideoStreamer",
        queue_size=10,
        stream="rtsp://admin:tp123456@192.168.10.221:554/Streaming/Channels/101",
    )

    def set_stream():
        time.sleep(5)
        streamer.reset_stream("rtsp://admin:tp123456@192.168.10.199:554/Streaming/Channels/101")

    cpipe_insight = CPipeInsight(http_insight=True)
    streamer += cpipe_insight

    thread = threading.Thread(target=set_stream)
    thread.start()

    Node.launch()