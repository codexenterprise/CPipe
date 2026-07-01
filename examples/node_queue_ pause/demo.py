from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer
from examples.node_queue_pause.custom_node import my_node


if __name__ == "__main__":

    streamer = VideoStreamer(node_name="streamer007", stream="/mnt/d/RT16000276624788CN_22_RI301.mp4")

    my_node = my_node()
    
    streamer += [my_node]

    Node.launch(check_node=True)
