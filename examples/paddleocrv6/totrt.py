from cpipe.module.model.paddleocr import PaddleOCRDet, PaddleOCRv6

# 在仓库根目录执行。需要已安装 tensorrt、cuda、cuDNN。
# PP-OCRv6_medium_det.onnx 输入名是 x，形状 NCHW，batch / 高 / 宽都是动态维。
# profile 与官方 inference.yml 一致，也和 PaddleOCRDet 对齐：高宽按 32 对齐，长边截到 4000。
# 引擎写到 ONNX 同目录：model_files/paddleocrv6/PP-OCRv6_medium_det.engine。
# 节点加载 .engine 时会按 max shape 预热，这里是 1x3x4000x4000。batch 保持 1，多路流会逐张跑。
PaddleOCRDet.onnx2tensorrt(
    "model_files/paddleocrv6/PP-OCRv6_medium_det.onnx",
    input_names=["x"],
    min_shapes=[(1, 3, 32, 32)],
    opt_shapes=[(1, 3, 736, 736)],
    max_shapes=[(1, 3, 4000, 4000)],
    fp16_mode=True,
)

# PP-OCRv6_medium_rec.onnx 输入名是 x，高固定 48，batch 和宽是动态维。
# profile 用官方 inference.yml。节点默认把文本行垫成 3x48x320，和 opt 一致。
# 一帧里多行文字走 batch，超过 8 行时 Cmodel 会拆开。字典用 ppocrv6_dict.txt。
# 引擎：model_files/paddleocrv6/PP-OCRv6_medium_rec.engine。
PaddleOCRv6.onnx2tensorrt(
    "model_files/paddleocrv6/PP-OCRv6_medium_rec.onnx",
    input_names=["x"],
    min_shapes=[(1, 3, 48, 160)],
    opt_shapes=[(1, 3, 48, 320)],
    max_shapes=[(8, 3, 48, 3200)],
    fp16_mode=True,
)
