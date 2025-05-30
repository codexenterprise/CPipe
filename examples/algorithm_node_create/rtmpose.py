from cpipe.module.model.rtmpose import RTMPose
from cpipe.module.model.yolov8 import YOLOv8obb
from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":
    stream = VideoStreamer("stream", "rtmp://192.168.8.121:1935/live/7777", 3, 1)

    zhu_det = YOLOv8obb("zhu_det",
                        "./models/VA.om",
                        3,
                        (3, 640, 640),
                        max_batch_size=1,
                        class_names=['开关座', '手', '接线柱红', '接线柱黑', '滑动变阻器', '滑片', '电压表', '电流表', '电源', '电阻'],
                        conf_thres=0.5, iou_thres=0.5,
                        )
    rp = RTMPose(
        "rp",
        "./models/hand-end2end_batch2.om",
        3,
        (3, 256, 256),
        max_batch_size=2,
        class_names=["pose"],
        secondary_class_names=["手"], # <<<<
    )

    cp = CPipeInsight(http_insight=True)

    stream += [zhu_det, rp, cp]

    Node.launch()
