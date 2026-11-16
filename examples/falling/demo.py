from cpipe.module.insight import CPipeInsight
from cpipe.module.model.stgcn import STGCN
from cpipe.module.model.tracker.tracker import Tracker
from cpipe.module.model.yolo26 import YOLO26Pose
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer


class PersonPose(YOLO26Pose):
    """Mark every pose box as person so the tracker can follow it."""

    def to_cdata(self, pred, new_cdata, frames, streamer_names, *args, **kwargs):
        super().to_cdata(pred, new_cdata, frames, streamer_names, *args, **kwargs)
        for streamer_name in streamer_names:
            for box in new_cdata.bboxes.get(streamer_name, []):
                box.box_class_name = "person"


if __name__ == "__main__":
    # 在仓库根目录运行。凑满 100 帧后才写出动作类别。
    stream = VideoStreamer(
        stream="project/fall_down/test.mp4",
        process_frame_interval=0,
    )
    pose = PersonPose(
        model_path="model_files/yolo26n-pose.engine",
        class_names=[
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ],
        conf_thres=0.35,
    )
    tracker = Tracker(
        tracker_type="ocsort",
        secondary_class_names=["person"],
        config_tracker={
            "det_thresh": 0.35,
            "max_age": 30,
            "min_hits": 1,
            "iou_threshold": 0.2,
            "delta_t": 2,
            "use_byte": True,
            "inertia": 0.2,
        },
    )
    action = STGCN(
        model_path="model_files/stgcnpp_ntu60_cpipe.onnx",
        class_names="model_files/stgcnpp_ntu60_cpipe.txt",
        window_size=100,
        clip_len=100,
        infer_interval=1,
        topk=1,
    )
    insight = CPipeInsight(http_insight=True, show_key_points=True)

    stream += [pose, tracker, action, insight]
    Node.launch(check_node=True, auto_restart=False)
