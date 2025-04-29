from cpipe_nodes.custom_node import my_node

from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    stream = VideoStreamer("vs1", "rtsp://admin:tp123456@192.168.8.220:554/Streaming/Channels/101", 3, 0)

    mn = my_node("text_show", 3)

    cpipeinsight = CPipeInsight(http_insight=True)

    stream += [mn, cpipeinsight]

    Node.launch(check_node=True, auto_restart=True)

