from cpipe.module.model.stgcn import STGCN

# 需要安装 tensorrt、cuda、cuDNN。在仓库根目录执行。
# 引擎与 ONNX 同目录，后缀为 .engine。输入名为 poses，形状为 [batch, 3, 100, 17]。

STGCN.onnx2tensorrt(
    "model_files/stgcnpp_ntu60_cpipe.onnx",
    input_names=["poses"],
    min_shapes=[(1, 3, 100, 17)],
    opt_shapes=[(16, 3, 100, 17)],
    max_shapes=[(16, 3, 100, 17)],
)
