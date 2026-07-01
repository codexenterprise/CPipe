from cpipe.module.model.paddleocr import PaddleOCR
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


if __name__ == "__main__":

    video_zhu = "/mnt/d/videos/dianhan/192.168.1.71_01_20250529164135113.mp4"

    stream_zhu = VideoStreamer(node_name="zhushitu", stream=video_zhu, process_frame_interval=1, once_mode=False)

    detect = YOLOv10(node_name="YOLOv10",
                     model_path="../../models/yolov10n.onnx",
                     input_size=(3, 640, 640),
                    #  class_names=['person', ...],
                     max_batch_size=1,
                     save_top_n_objects=32,
                     area_flag=True
                     )

    ocr = PaddleOCR(
        node_name="PaddleOCR",
        model_path="/home/zhouhe/workspace/cpipe2.0/project/dianhan/models/OCRv4.onnx",
        queue_size=3,
        keys_txt_path="/home/zhouhe/workspace/cpipe2.0/project/dianhan/models/ppocr_keys_v1.txt",
        max_batch_size=1,
        secondary_class_names=["stop sign"]
    )

    cpipeinsight = CPipeInsight(http_insight=True)

    stream_zhu += [detect, ocr, cpipeinsight]

    Node.launch(check_node=True)