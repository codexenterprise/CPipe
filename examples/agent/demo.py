from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from custom_node import my_node 
if __name__ == "__main__":

    stream = VideoStreamer(node_name="vs1", stream="./text.mp4", process_frame_interval=3)

    mn = my_node(node_name="text_show", queue_size=3)

    cpipeinsight = CPipeInsight(http_insight=True)

    stream += [mn, cpipeinsight]

    Node.launch(check_node=True, auto_restart=True, agent=True)
