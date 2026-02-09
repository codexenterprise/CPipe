from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from examples.personnel_intrusion.custom_node import PersonnelIntrusion

if __name__ == "__main__":
    stream1 = VideoStreamer(stream="rtmp://192.168.10.7:1935/live/7777", process_frame_interval=1, once_mode=True)
    detect = YOLOv10(
                    model_path="model_files/yolov10n_batch1.engine",
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
                    area_flag=True
                    )

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=2, save_video=True, chinese_font_size=30)

    personnel_intrusion = PersonnelIntrusion()

    stream1 += [detect, personnel_intrusion, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False, agent=False)
