
import time
from cpipe.module.node import Node
import threading
from cpipe.module.streamer import VideoStreamer


if __name__ == "__main__":
    

    streamer = VideoStreamer(
        nodeName="VideoStreamer",
        queue_size=10,
        stream="rtsp://192.168.1.100:8554/test",
    )
    def set_stream():
        time.sleep(5)
        streamer.reset_stream("rtsp://192.168.1.100:8554/test")


    thread = threading.Thread(target=set_stream)
    thread.start()

    Node.launch()