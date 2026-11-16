from cpipe.module.insight import CPipeInsight
from cpipe.module.model.qrcode import QRCode
from cpipe.module.model.yolov8 import YOLOv8
from cpipe.module.node import Node
from cpipe.module.streamer import ImageStreamer, VideoStreamer


if __name__ == "__main__":
    # 在仓库根目录运行。output_wh 用相机分辨率，不要先把大图缩小。
    stream = VideoStreamer(
        stream="rtsp://admin:tp123456@192.168.10.221:554/Streaming/Channels/101",
    )
    detector = YOLOv8(
        model_path="model_files/QR_yolov8.engine",
    )
    qr = QRCode(
        crop_padding=0.1,
        min_side=320,
    )
    insight = CPipeInsight(http_insight=True, show_scale=1)

    stream += [detector, qr, insight]
    Node.launch()
