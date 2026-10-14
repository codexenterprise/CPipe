from cpipe.module.insight import CPipeInsight
from cpipe.module.model.paddleocr import PaddleOCRDet, PaddleOCRv6
from cpipe.module.node import Node
from cpipe.module.streamer import ImageStreamer


if __name__ == "__main__":
    stream = ImageStreamer(
        stream="/mnt/d/cars/3",
        output_wh=(1440, 1080),
        once_mode=False,
        play_interval=3,
    )

    det = PaddleOCRDet(
        model_path="model_files/paddleocrv6/PP-OCRv6_medium_det.onnx",
    )
    # small / medium 共用 ppocrv6_dict.txt。tiny 要换成 ppocrv6_tiny_dict.txt。
    ocr = PaddleOCRv6(
        keys_txt_path="model_files/paddleocrv6/ppocrv6_tiny_dict.txt",
        model_path="model_files/paddleocrv6/PP-OCRv6_tiny_rec.onnx",
        secondary_class_names=["text"],
        rotate_tall=True,
    )

    insight = CPipeInsight(http_insight=True, show_scale=1.0)
    stream += [det, ocr, insight]
    Node.launch(check_node=True)
