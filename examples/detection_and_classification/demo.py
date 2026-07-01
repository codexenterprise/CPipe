from cpipe.module.model.shufflenet import MMShuffleNet
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from cpipe.module.model.shufflenet import MMShuffleNet

if __name__ == "__main__":
    stream1 = VideoStreamer(stream="rtsp://admin:tp123456@192.168.10.199:554/Streaming/Channels/101", process_frame_interval=0, once_mode=True)
    detect = YOLOv10(
                    model_path="model_files/yolov10n.onnx",
                    # class_names=['person', ...],
                    valid_class_names=["person"],
                    save_top_n_objects=32,
                    area_flag=False
                    )
    cls_model = MMShuffleNet(
                    model_path="/mnt/d/model.onnx.cpipe",
                    class_names=[["vest_ok", "vest_ng"], ["helmet_ok", "helmet_ng"]],
                    secondary_class_names=["person"],
                    )

    cpipeinsight = CPipeInsight(http_insight=True)

    stream1 += [detect, cls_model, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False)
