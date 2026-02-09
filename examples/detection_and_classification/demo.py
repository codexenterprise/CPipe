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
                    class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                                'hair drier', 'toothbrush'],
                    valid_class_names=["person"],
                    save_top_n_objects=32,
                    area_flag=False
                    )
    cls_model = MMShuffleNet(  # pyright: ignore[reportUndefinedVariable]
                    model_path="/mnt/d/model.onnx.cpipe",
                    class_names=[["vest_ok", "vest_ng"], ["helmet_ok", "helmet_ng"]],
                    secondary_class_names=["person"],
                    )

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=1, save_video=True, chinese_font_size=30, save_fps=2)



    stream1 += [detect, cls_model, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False, agent=False)
