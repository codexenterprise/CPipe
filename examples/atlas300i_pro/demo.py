import numpy as np

from cpipe.module.cinferencehook import HKI_CropImage, HKI_DilateImage
from cpipe.module.model.movenet import MoveNet
from cpipe.module.model.rtmpose import RTMPose
from cpipe.module.model.shufflenet import ShuffleNet
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.model.yolov8 import YOLOv8obb
from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":
    stream = VideoStreamer("stream", "./PAPER_20250331141736_20210204_D5.mp4", 3, 0)
    # stream = VideoStreamer("stream", "rtmp://192.168.8.121:1935/live/7777", 3, 1)

    zhu_det = YOLOv8obb("zhu_det",
                        "./models/VA.om",
                        3,
                        (3, 640, 640),
                        max_batch_size=1,
                        class_names=['开关座', '手', '接线柱红', '接线柱黑', '滑动变阻器', '滑片', '电压表', '电流表', '电源', '电阻'],
                        conf_thres=0.5, iou_thres=0.5,
                        device="npu:0"
                        )

    # zhu_det = YOLOv7("YOLOv7",
    #                 #  "./models/yolov7-tiny_rknn_batch4.om",
    #                  "./models/yolov7-tiny_rknn_om_batch4.om",
    #                  3,
    #                  (3, 640, 640),
    #                  class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    #                               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    #                               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #                               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    #                               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #                               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #                               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    #                               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    #                               'hair drier', 'toothbrush'],
    #                  max_batch_size=4,
    #                  valid_class_names=["person"],
    #                  save_top_n_objects=32,
    #                  area_flag=True,
    #                  device="npu:0",
    #                  anchor=np.array([12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0]).reshape(3, -1, 2).tolist()
    #                  )

    xian = ShuffleNet("xian",
                      "./models/zhuzi_4.18_dct_batch32.om",
                      3,
                      inputSize=(3, 96, 96),
                      class_names=[['NG', 'OK']],
                      max_batch_size=32,
                      warmup=True,
                      secondary_class_names=['接线柱红', '接线柱黑'],
                      device="npu:0",
                      hook_inputs=HKI_DilateImage([1 / 1.7, 1 / 1.9])
                      )

    kaiguanzuo = ShuffleNet("kaiguanzuo",
                            # "./src/dengpao/kaiguan_11_28_gray_batch64.engine",
                            "./models/kaiguan_11_29_128x192_gray_batch4.om",
                            3,
                            # inputSize=(3, 128, 128),
                            inputSize=(3, 128, 192),
                            class_names=[['开', "未知", '合']],
                            max_batch_size=2,
                            warmup=True,
                            device="npu:0",
                            secondary_class_names=['开关座'],
                            gray_mode=True
                            )

    dianbiao = MoveNet(
        "dianbiao",
        "./models/dianbiao_192_4.30_batch3.om",
        3,
        (3, 192, 192),
        class_names=['1', "2", "3", "4", "5", "6"],
        center_weight_path="./models/center_weight_origin.npy",
        num_joints=6,
        max_batch_size=3,
        secondary_class_names=['电流表', '电压表'],
        hook_inputs=HKI_CropImage([0.0, 0.5, 0.0, 1.0]),
        device="npu:0"
    )

    rp = RTMPose(
        "rp",
        "./models/hand-end2end_batch2.om",
        3,
        (3, 256, 256),
        max_batch_size=2,
        class_names=["pose"],
        secondary_class_names=["手"],
        device="npu:0"
    )

    cp = CPipeInsight(http_insight=True)

    stream += [zhu_det, xian, kaiguanzuo, dianbiao, rp, cp]

    Node.launch()
