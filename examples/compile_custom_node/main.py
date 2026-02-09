from cpipe_nodes.custom_node import my_node

from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    stream = VideoStreamer(node_name="vs1", stream="rtsp://admin:tp123456@192.168.8.220:554/Streaming/Channels/101", process_frame_interval=1, pause=0)

    mn = my_node(node_name="text_show", queue_size=3)

    cpipeinsight = CPipeInsight(http_insight=True)

    stream += [mn, cpipeinsight]

    Node.launch(check_node=True, auto_restart=True)

