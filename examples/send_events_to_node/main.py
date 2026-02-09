import threading
import time
import numpy as np

from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from custom_node import my_node


def one_thread():
    i = 0
    while True:
        ret = Node.__allNodes__["text_show"].event_send("print_event", {"msg": f"hello world{i}", "ndarray": np.array([1, 2, 3])})
        print("event_send ret:", ret)
        time.sleep(1)
        i += 1

if __name__ == "__main__":
    stream = VideoStreamer(stream="rtmp://192.168.8.7:1935/live/7777", process_frame_interval=1)

    mn = my_node(node_name="text_show", queue_size=3)

    t = threading.Thread(target=one_thread, daemon=True)
    t.start()

    cpipeinsight = CPipeInsight(http_insight=True)

    stream += [mn, cpipeinsight]

    Node.launch(check_node=True, auto_restart=True)
