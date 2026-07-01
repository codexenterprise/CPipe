from cpipe.module.insight import CPipeInsight
from cpipe.module.model.yolov11 import YOLOv11Pose
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":
    stream = VideoStreamer(stream="rtmp://192.168.10.7:1935/live/7777")
    stream1 = VideoStreamer(stream="rtmp://192.168.10.7:1935/live/7777")

    # pose = YOLOv11Pose(model_path="/mnt/d/yolo11x-pose.onnx")
    pose = YOLOv11Pose(model_path="/mnt/d/yolo11x-pose_batch1.engine")

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=2, show_key_points_name=True, save_video=True)

    stream += [pose, cpipeinsight]
    stream1 += [pose, cpipeinsight]
    Node.launch()