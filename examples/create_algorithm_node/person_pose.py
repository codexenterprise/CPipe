from cpipe.module.insight import CPipeInsight
from cpipe.module.model.movenet import MoveNetPersonPose
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":
    video_zhu = "rtmp://192.168.8.121:1935/live/7777"
    queue_size = 3

    stream_zhu = VideoStreamer(node_name="zhushitu", stream=video_zhu, process_frame_interval=1, once_mode=True)

    det = YOLOv7(node_name="YOLOv7",
                model_path="./models/qxj.engine",
                input_size=(3, 640, 640),
                max_batch_size=1,
                class_names=['人'],
                )

    person = MoveNetPersonPose(node_name="person_pose",
                                model_path="./models/movenet_person_pose.onnx",
                                input_size=(3, 256, 256),
                                num_joints=17,
                                secondary_class_names=["人"]
                                )

    cpipeinsight = CPipeInsight(http_insight=True, port=9966)

    stream_zhu += [det, person, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False, check_interval=5)
