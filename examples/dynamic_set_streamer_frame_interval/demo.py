from cpipe.module.streamer import VideoStreamer


if __name__ == "__main__":
    vs = VideoStreamer(node_name="streamers", stream=0, process_frame_interval=1)
    # streamer dynamic set process_frame_interval
    vs.process_frame_interval(1)

