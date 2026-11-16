from cpipe.module.model.yolov8 import YOLOv8

# 需要安装 tensorrt、cuda、cuDNN。在仓库根目录执行。
# best-yolov8.onnx 是静态输入 images: 1x3x640x640，三个 profile 保持一致。
# 引擎与 ONNX 同目录，文件名为 best-yolov8.engine。

YOLOv8.onnx2tensorrt(
    "model_files/QR_yolov8.onnx",
    input_names=["images"],
    min_shapes=[(1, 3, 640, 640)],
    opt_shapes=[(4, 3, 640, 640)],
    max_shapes=[(4, 3, 640, 640)],
)
