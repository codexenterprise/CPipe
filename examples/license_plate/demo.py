from cpipe.module.insight import CPipeInsight
from cpipe.module.model.paddleplate import PaddlePlate
from cpipe.module.model.yolo26 import YOLO26
from cpipe.module.node import Node
from cpipe.module.streamer import ImageStreamer


if __name__ == "__main__":
    stream = ImageStreamer(
        output_wh=(4096, 3072),
        stream="/mnt/d/cars/1",
        once_mode=False,
        play_interval=3,
    )

    vehicle = YOLO26(
        node_name="YOLO26",
        model_path="model_files/yolo26m.engine",
        valid_class_names=["car", "bus", "truck"],
        save_top_n_objects=64,
    )

    plate = PaddlePlate(
        node_name="PaddlePlate",
        det_model_path="model_files/paddleplate/ch_PP-OCRv3_det.engine",
        rec_model_path="model_files/paddleplate/ch_PP-OCRv3_rec.engine",
        keys_txt_path="src/model_files/ppocr_keys_v1.txt",
        secondary_class_names=["car", "bus", "truck"],
        det_limit_side_len=736,
        det_limit_type="min",
        area_flag=False,
    )

    insight = CPipeInsight(http_insight=True, show_scale=1.0, chinese_font_size=90)
    stream += [vehicle, plate, insight]
    Node.launch(check_node=True)
