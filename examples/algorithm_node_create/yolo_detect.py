from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.model.yolov11 import YOLOv11
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    stream = VideoStreamer(node_name="stream", stream="rtmp://192.168.8.122:1935/live/7777", process_frame_interval=3)

    detect = YOLOv10(node_name="YOLOv10",
                     model_path="../../models/yolov10n.onnx",
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

    cp = CPipeInsight(http_insight=True)
    stream += [detect, cp]

    Node.launch()
