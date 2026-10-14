from cpipe.module.model.paddleplate import PaddlePlate

# 先删掉旧的 .engine 再跑这个脚本。识别引擎如果还是静态 batch=1，预热时喂 [8,3,48,320] 会报 Static dimension mismatch。
# 一帧里的多辆车按 batch 一起做车牌检测和识别。超过 8 辆时 Cmodel 会拆成多批。
# 检测高宽随车框变化，对齐到 32，长边在节点里截到 2048，和这里的 max_shapes 一致。
PaddlePlate.onnx2tensorrt(
    "model_files/paddleplate/ch_PP-OCRv3_det.onnx",
    input_names=["x"],
    min_shapes=[(1, 3, 32, 32)],
    opt_shapes=[(8, 3, 736, 736)],
    max_shapes=[(8, 3, 2048, 2048)],
)

# 识别在节点里固定成 3x48x320，多块车牌放在 batch 维。
PaddlePlate.onnx2tensorrt(
    "model_files/paddleplate/ch_PP-OCRv3_rec.onnx",
    input_names=["x"],
    min_shapes=[(1, 3, 48, 320)],
    opt_shapes=[(8, 3, 48, 320)],
    max_shapes=[(8, 3, 48, 320)],
)
