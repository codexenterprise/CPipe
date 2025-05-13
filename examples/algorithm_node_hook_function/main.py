from cpipe.module.model.yolov8 import YOLOv8obb
from my_hooks import HKI_CropImage, HKI_DilateImage, HKO_ClassNamesThresholdFilter
from cpipe.module.model.movenet import MoveNet
from cpipe.module.model.shufflenet import ShuffleNet


if __name__ == "__main__":
    zhu_det = YOLOv8obb("zhu_det",
                        # "./src/dengpao/dengpao_zhu11.12_batch1.engine",
                        "project/dct/models/dct_batch.engine",
                        3,
                        (3, 640, 640),
                        max_batch_size=1,
                        class_names=['大电磁铁', '小电磁铁', '开关座', '手', '接线柱红', "接线柱黑", '滑动变阻器', '滑片', '电流表', '电源', "钉子", "钉盒"],
                        conf_thres=0.5, iou_thres=0.5,

                        # hook_outputs=HKO_DumpClass(-1, ["钉盒"]), # <<<<<<<<<<<<<<<<<<< hook output mode
                        hook_outputs=HKO_ClassNamesThresholdFilter(-1, -2, {"大电磁铁": 0.75}), # <<<<<<<<<<<<<<<<<<< hook output mode
                        )


    xian = ShuffleNet("xian",
                      "project/fuanfa/models/zhuzi_1_6_batch32.engine",
                      3,
                      inputSize=(3, 96, 96),
                      class_names=[['OK', 'NG']],
                      max_batch_size=32,
                      warmup=True,
                      secondary_class_names=['接线柱红', '接线柱黑'],
                      device="cuda:0",
                      hook_inputs=HKI_DilateImage([1/1.7, 1/1.9])  # <<<<<<<<<<<<<<<<<<< hook input mode
                      )

    dianbiao = MoveNet(
        "dianbiao",
        "project/fuanfa/models/dianbiao_192_4.30.engine",
        3,
        (3, 192, 192),
        # (3, 320, 320),
        class_names=['1', "2", "3", "4", "5", "6"],
        center_weight_path="project/fuanfa/models/center_weight_origin.npy",
        num_joints=6,
        max_batch_size=2,
        secondary_class_names=['电压表', '电流表'],
        hook_inputs=HKI_CropImage([0.0, 0.5, 0.0, 1.0])  # <<<<<<<<<<<<<<<<<<< hook input mode
    )

