import numpy as np
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":
    stream = VideoStreamer("stream", "../../../PAPER_20250331141736_20210204_D5.mp4", 3, 0)


    zhu_det = YOLOv7("YOLOv7",
                     "../../../models/yolov7-tiny_rknn_om_batch4.om",
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
                     max_batch_size=4,
                     valid_class_names=["person"],
                     save_top_n_objects=32,
                     area_flag=True,
                     device="npu:0",
                     anchor=np.array([12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0]).reshape(3, -1, 2).tolist()
                     )


    cp = CPipeInsight(http_insight=True)

    stream += [zhu_det, cp]

    Node.launch()
