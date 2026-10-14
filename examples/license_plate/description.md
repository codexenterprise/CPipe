用 PaddleDetection 2.8.1 PP-Vehicle 的车牌识别（PP-OCRv3 检测 + PP-OCRv3 识别）读车牌号。

先用 `export_onnx.py` 把官方推理模型转成 ONNX。转换需要 PaddlePaddle 2.6 和 paddle2onnx 1.2，PaddlePaddle 3 导出会失败。运行节点还需要 `pyclipper`。

示例先用 YOLO26 检车，再在车框里读车牌，号码写回车辆框的 `box_text`。`totrt.py` 按 CPipe 的 `onnx2tensorrt` 把检测和识别 ONNX 转成 TensorRT，引擎文件与 ONNX 同名、后缀为 `.engine`。
