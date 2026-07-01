from cpipe.module.insight import CPipeInsight
from cpipe.module.model.yolov8 import YOLOv8obb
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer
from my_hooks import HKI_CropImage, HKI_DilateImage, HKO_ClassNamesThresholdFilter
from cpipe.module.model.movenet import MoveNet
from cpipe.module.model.shufflenet import ShuffleNet


if __name__ == "__main__":
    stream = VideoStreamer(node_name="stream", stream="rtmp://192.168.8.121:1935/live/7777", process_frame_interval=1)

    zhu_det = YOLOv8obb(node_name="zhu_det",
                        model_path="project/dct/models/dct_batch.engine",
                        input_size=(3, 640, 640),
                        max_batch_size=1,
                        class_names=['大电磁铁', '小电磁铁', '开关座', '手', '接线柱红', "接线柱黑", '滑动变阻器', '滑片', '电流表', '电源', "钉子", "钉盒"],
                        conf_thres=0.5, iou_thres=0.5,

                        # hook_outputs=HKO_DumpClass(-1, ["钉盒"]), # <<<<<<<<<<<<<<<<<<< hook output mode
                        hook_outputs=HKO_ClassNamesThresholdFilter(-1, -2, {"大电磁铁": 0.75}), # <<<<<<<<<<<<<<<<<<< hook output mode
                        )


    xian = ShuffleNet(node_name="xian",
                      model_path="project/fuanfa/models/zhuzi_1_6_batch32.engine",
                      input_size=(3, 96, 96),
                      class_names=[['OK', 'NG']],
                      warmup=True,
                      secondary_class_names=['接线柱红', '接线柱黑'],
                      device="cuda:0",
                      hook_inputs=HKI_DilateImage([1/1.7, 1/1.9])  # <<<<<<<<<<<<<<<<<<< hook input mode
                      )

    dianbiao = MoveNet(
        node_name="dianbiao",
        model_path="project/fuanfa/models/dianbiao_192_4.30.engine",
        input_size=(3, 192, 192),
        # (3, 320, 320),
        class_names=['1', "2", "3", "4", "5", "6"],
        center_weight_path="project/fuanfa/models/center_weight_origin.npy",
        num_joints=6,
        max_batch_size=2,
        secondary_class_names=['电压表', '电流表'],
        hook_inputs=HKI_CropImage([0.0, 0.5, 0.0, 1.0])  # <<<<<<<<<<<<<<<<<<< hook input mode
    )

    cp = CPipeInsight(http_insight=True)
    zhu_det += [xian, cp]

    Node.launch()

