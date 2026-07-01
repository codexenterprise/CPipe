from cpipe.module.model.yolov8 import YOLOv8obb
from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    stream = VideoStreamer(node_name="stream", stream="rtmp://192.168.8.122:1935/live/7777", process_frame_interval=1)

    obb = YOLOv8obb(node_name="zhu_det",
                        model_path="./models/VA.om",
                        input_size=(3, 640, 640),
                        max_batch_size=1,
                        class_names=['开关座', '手', '接线柱红', '接线柱黑', '滑动变阻器', '滑片', '电压表', '电流表', '电源', '电阻'],
                        conf_thres=0.5, iou_thres=0.5,
                        )

    cp = CPipeInsight(http_insight=True)
    stream += [obb, cp]

    Node.launch()
