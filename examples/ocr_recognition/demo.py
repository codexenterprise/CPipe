from cpipe.module.cinferencehook import HKI_CropImage
from cpipe.module.clogger import CLogger
from cpipe.module.model.movenet import MoveNet
from cpipe.module.model.paddleocr import PaddleOCR
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from project.dianhan.dianhan import DianHan


if __name__ == "__main__":

    video_zhu = "/mnt/d/videos/dianhan/192.168.1.71_01_20250529164135113.mp4"

    stream_zhu = VideoStreamer("zhushitu", video_zhu, 3, 1, once_mode=False)

    detect = YOLOv10("YOLOv10",
                     "../../models/yolov10n.onnx",
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
                     save_top_n_objects=32,
                     area_flag=True
                     )

    ocr = PaddleOCR(
        "PaddleOCR",
        "/home/zhouhe/workspace/cpipe2.0/project/dianhan/models/OCRv4.onnx",
        3,
        keys_txt_path="/home/zhouhe/workspace/cpipe2.0/project/dianhan/models/ppocr_keys_v1.txt",
        max_batch_size=1,
        secondary_class_names=["stop sign"]
    )

    cpipeinsight = CPipeInsight(http_insight=True)

    stream_zhu += [detect, ocr, cpipeinsight]

    Node.launch(check_node=True)