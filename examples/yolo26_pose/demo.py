from cpipe.module.insight import CPipeInsight
from cpipe.module.model.yolo26 import YOLO26Pose
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":
    video = "/mnt/d/test.mp4"

    pose_stream = VideoStreamer(node_name="pose_stream", stream=video, process_frame_interval=1, block_mode=True)

    pose = YOLO26Pose(
        node_name="YOLO26Pose",
        model_path="model_files/yolo26n-pose.engine",
        conf_thres=0.25,
    )

    insight = CPipeInsight(http_insight=True, show_mask=True, show_polygon=True, show_key_points=True)

    pose_stream += [pose, insight]
    Node.launch(check_node=True, auto_restart=False)
