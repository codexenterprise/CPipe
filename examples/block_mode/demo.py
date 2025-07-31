from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


if __name__ == "__main__":
    # stream = VideoStreamer("ss", "rtsp://admin:tp123456@192.168.8.199:554/Streaming/Channels/101", 3, 1)
    stream1 = VideoStreamer("ss1", "/mnt/d/videos/other/face_2.mp4", 3, 1)
    detect = YOLOv10("D1",
                    "/home/zhouhe/workspace/cpipe2.0/src/model_files/yolov10n.onnx",
                    3,
                    (3, 640, 640),
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

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=2)

    stream1 += [detect, cpipeinsight]

    # launch all initialized nodes
    Node.launch(check_node=True, auto_restart=False)
