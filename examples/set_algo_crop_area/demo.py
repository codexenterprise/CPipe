from cpipe.module.model.movenet import MoveNet
from cpipe.module.model.yolov8 import YOLOv8obb

if __name__ == "__main__":
    y = YOLOv8obb("YOLOv8obb",
                  "model.engine",
                  3,
                  (3, 640, 640),
                  max_batch_size=1,
                  class_names=['开关座', '手', '接线柱红', '接线柱黑', '滑动变阻器', '滑片', '电压表', '电流表', '电源', '电阻'],
                  conf_thres=0.5, iou_thres=0.5,
                  # crop_factor: (list) The crop factor of the input image. (h start(0~1), h end(0~1), w start(0~1), w end(0~1)).
                  crop_factor=[0.0, 0.5, 0.0, 1.0],  # <<<<<<<<<<<<<<<<<<<<<<<<<< set algo crop area
                  )

    m = MoveNet(
        "MoveNet",
        "model.onnx",
        3,
        (3, 192, 192),
        class_names=['1', "2", "3", "4", "5", "6"],
        center_weight_path="project/fuanfa/models/center_weight_origin.npy",
        num_joints=6,
        max_batch_size=2,
        secondary_class_names=['电压表', '电流表'],
        # crop_factor: (list) The crop factor of the input image. (h start(0~1), h end(0~1), w start(0~1), w end(0~1)).
        crop_factor=[0.0, 0.5, 0.0, 1.0],  # <<<<<<<<<<<<<<<<<<<<<<<<<< set algo crop area
    )
