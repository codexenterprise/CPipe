from cpipe.module.model.yolo26 import YOLO26InstanceSeg, YOLO26Pose

# 需要安装 tensorrt、cuda、cuDNN。在仓库根目录执行。
# ONNX 须用端到端导出：yolo export model=yolo26n-seg.pt format=onnx dynamic=True imgsz=640 opset=19 nms=False
# 引擎与 ONNX 同目录，后缀为 .engine。

YOLO26Pose.onnx2tensorrt(
    "model_files/yolo26n-pose.onnx",
    input_names=["images"],
    min_shapes=[(1, 3, 640, 640)],
    opt_shapes=[(1, 3, 640, 640)],
    max_shapes=[(1, 3, 640, 640)],
)
