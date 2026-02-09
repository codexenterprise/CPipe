import threading
import time

import cv2
from cpipe.module.model.tracker.tracker import Tracker
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":

    def get_result(cpipeinsight):
        while True:
            time.sleep(0.04)
            result = cpipeinsight.get_current_show_image("streamer1")
            if result is not None:
                cv2.imshow("result", result)
                cv2.waitKey(1)
                # print(result.shape)

    # 视频流模式
    streamer1 = VideoStreamer("streamer1", "rtmp://192.168.10.7:1935/live/7777", 3, 1)

    detect = YOLOv10("YOLOv10",
                     "src/model_files/yolov10n.onnx",
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

    cpipeinsight = CPipeInsight(http_insight=True, save_video=True)  # save_video必须为True

    streamer1 += [detect, cpipeinsight]

    t = threading.Thread(target=get_result, args=(cpipeinsight,))
    t.start()

    Node.launch(check_node=True, check_interval=5, auto_restart=False)  # auto_restart必须为False
