from cpipe.module.insight import CPipeInsight
from cpipe.module.model.yolo26 import YOLO26InstanceSeg
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":
    video = "/mnt/d/test.mp4"

    seg_stream = VideoStreamer(node_name="seg_stream", stream=video, process_frame_interval=1, block_mode=True)

    seg = YOLO26InstanceSeg(
        node_name="YOLO26InstanceSeg",
        model_path="model_files/yolo26n-seg.engine",
        return_mask=True,
        conf_thres=0.25,
    )

    insight = CPipeInsight(http_insight=True, show_mask=True, show_polygon=True, show_key_points=True)

    seg_stream += [seg, insight]
    Node.launch(check_node=True, auto_restart=False)
