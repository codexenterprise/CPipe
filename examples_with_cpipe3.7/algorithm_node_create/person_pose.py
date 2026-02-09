from cpipe.module.insight import CPipeInsight
from cpipe.module.model.movenet import MoveNetPersonPose
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":


    video_zhu = "rtmp://192.168.8.121:1935/live/7777"
    queue_size = 3

    stream_zhu = VideoStreamer("zhushitu", video_zhu, 3, 1, once_mode=True)

    det = YOLOv7("YOLOv7",
                        "./models/qxj.engine",
                        queue_size,
                        (3, 640, 640),
                        max_batch_size=1,
                        class_names=['人'],
                        )

    person = MoveNetPersonPose("person_pose",
                                      "./models/movenet_person_pose.onnx",
                                      3, [3, 256, 256],
                                      [i for i in range(17)], 17,
                                      secondary_class_names=["人"]
                                      )

    cpipeinsight = CPipeInsight(http_insight=True, port=9966)

    stream_zhu += [det, person, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False, check_interval=5)
