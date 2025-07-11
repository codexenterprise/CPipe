from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer



if __name__ == "__main__":

    vs = VideoStreamer("streamers", 0, 3)
    # streamer dynamic set process_frame_interval
    vs.process_frame_interval(1)

