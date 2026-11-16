from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


if __name__ == "__main__":
    stream = VideoStreamer(node_name="stream", stream="./test.mp4", process_frame_interval=1)
    cpipeinsight = CPipeInsight(
        http_insight=True,
        show_box=False,
        show_box_name=False,
        show_polygon=False,
    )
    stream += cpipeinsight
    Node.launch(check_node=True, auto_restart=False)

