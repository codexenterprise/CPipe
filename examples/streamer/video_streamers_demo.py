from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamers

if __name__ == "__main__":

    # local USB camera
    vs1 = VideoStreamers(node_name="streamers", streams=[
        "rtsp://admin:tp123456@192.168.10.5:554/Streaming/Channels/101", 
        "rtsp://admin:tp123456@192.168.10.998:554/Streaming/Channels/101", 
        "rtsp://admin:tp123456@192.168.10.7:554/Streaming/Channels/101", 
    ], 
    processor_num=3,
    interval_time=0., # After each stream is completed, wait for the interval_time before starting to pull the next stream.
    round_interval_time=0., # After all the videos in the streams have completed their rotation, wait for the duration of the "round_interval_time" and then continue to re-rotate.
    device="cuda:0") 

        # local USB camera
    vs2 = VideoStreamers(node_name="streamers", streams=[
        "rtsp://admin:tp123456@192.168.10.119:554/Streaming/Channels/101", 
        "rtsp://admin:tp123456@192.168.10.221:554/Streaming/Channels/101", 
        "rtsp://admin:tp123456@192.168.10.184:554/Streaming/Channels/101", 
    ], 
    processor_num=1,
    interval_time=0., # After each stream is completed, wait for the interval_time before starting to pull the next stream.
    round_interval_time=0., # After all the videos in the streams have completed their rotation, wait for the duration of the "round_interval_time" and then continue to re-rotate.
    device="cuda:0") 

    cp = CPipeInsight(http_insight=True)
    vs1 += cp
    vs2 += cp

    Node.launch()
