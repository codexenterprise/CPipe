from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from custom_node import my_node
if __name__ == "__main__":

    stream = VideoStreamer("vs1", "./text.mp4", 3, 1)

    mn = my_node("text_show", 3)

    cpipeinsight = CPipeInsight(http_insight=True)

    stream += [mn, cpipeinsight]

    Node.launch(check_node=True, auto_restart=True)
