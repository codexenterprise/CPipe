from cpipe.module.insight import CPipeInsight
from cpipe.module.model.movenet import MoveNetPersonPose
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":


    video_zhu = "rtsp://admin:tp123456@192.168.10.221:554/Streaming/Channels/101"
    queue_size = 3

    stream_zhu1 = VideoStreamer(node_name="zhushitu1", stream=video_zhu, process_frame_interval=1, once_mode=True)
    stream_zhu2 = VideoStreamer(node_name="zhushitu2", stream=video_zhu, process_frame_interval=1, once_mode=True)
    stream_zhu3 = VideoStreamer(node_name="zhushitu3", stream=video_zhu, process_frame_interval=1, once_mode=True)

    detect = YOLOv10(
                     model_path="model_files/yolov10n_batch1.engine",
                     input_size=(3, 640, 640),
                     class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                                  'hair drier', 'toothbrush'],
                     max_batch_size=1,
                     valid_class_names=["person"],
                     save_top_n_objects=32,
                     area_flag=True
                     )

    person = MoveNetPersonPose(node_name="person_pose",
                                      model_path="model_files/movenet_person_pose_batch1.engine",
                                      secondary_class_names=["person"]
                                      )

    cpipeinsight = CPipeInsight(http_insight=True, port=9966)

    stream_zhu1 += [detect, person, cpipeinsight]
    stream_zhu2 += [detect, person, cpipeinsight]
    stream_zhu3 += [detect, person, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False, check_interval=5)
