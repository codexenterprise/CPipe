## 这套模型是什么

Decider 是一次前向的决策模型（不是生成式 VLM）：对画面问 noul / choice / score，读出 `<|fim_suffix|>` 位置的 logit。
官方 HuggingFace 产物是 **一份 PyTorch**（视觉 + Qwen3.5 LM + 255×2560 readout）。
接到 ONNX / TensorRT 时拆成 3 张图，**不是官方默认格式**：

| 文件 | 做什么 | 动态维 | 推荐后端 |
|---|---|---|---|
| `vision.onnx` / `vision_fp16.engine` | 图像 patch → image token 特征 | N（patch 数） | GPU，TRT FP16 |
| `embed.onnx` | `input_ids` → token 向量（词表 Gather） | T | **CPU ONNX**，不要转 TRT |
| `decoder.onnx` / `decoder_fp16.engine` | 序列 + readout → 每题 logit | 生产路径冻死 T、Q | GPU，TRT FP16 |

拆开的原因：vision 的 N 随分辨率变；embed 那张 248320×2560 表在 CPU Gather 约 0.1ms，上 GPU 更慢还占 2.4GB；decoder 里 Gated-DeltaNet 的 `Scan` 要 T 是 64 的倍数，TRT 对动态 T + Scan 不友好。

三张图不要合成一个 engine。4090 上 decoder 才是大头（~70ms），合成几乎不加速，Scatter+Scan 还容易导出失败。

## 分辨率和 T

节点 `input_size=[3,320,320]` 做 letterbox。320² → 20×20 patch = 400，vision merger 后约 100 个 image token，加上 prompt 后总长会超过 256。
decoder 的 T 必须是 `CHUNK=64` 的倍数，所以生产图冻 **T=320、Q=2**。

注意：Qwen 的 `head_dim=256` 和旧图的 `T=256` 碰巧同值，改 T 时不要把 head_dim 一起改掉。

## 环境

导出 ONNX 需要 **完整 HF ckpt**（约 8.7GB，含 `source/src/pplx_decider`），tokenizer-only 目录不够。

```text
# 4090 上实测环境（按本机路径改）
Python:  /media/workspace/uv-envs/py312/bin/python
工作目录: /media/workspace/zhouhe/decider-onnx
ckpt:    /media/workspace/zhouhe/decider-onnx/ckpt
TensorRT 10.15 + CUDA，转 engine 时 GPU 要空出约 16GB
# 若 GPU0 被 vLLM 占着：CUDA_VISIBLE_DEVICES=1
```

本目录脚本依赖同目录的 `qwen35_onnx.py`，在 `cpipe/examples/decider` 下执行。

## 流程 1：HF → ONNX

默认导出 vision（动态 N）+ embed（动态 T）+ decoder（静态 T=320、Q=2）：

```bash
cd cpipe/examples/decider

python export_onnx.py \
    --ckpt /media/workspace/zhouhe/decider-onnx/ckpt \
    --out  /media/workspace/zhouhe/decider-onnx/out-t320
```

只重导 decoder（vision/embed 已有时）：

```bash
python export_onnx.py --ckpt .../ckpt --out .../out-t320 --only decoder --T 320 --Q 2
```

动态 decoder（仅 ORT 实验，TRT 不要用；cpipe 的 Decider 节点只加载静态 batch / T / Q 的 decoder，这张图会在启动时报错）：

```bash
python export_onnx.py --ckpt .../ckpt --out .../out --dynamic-decoder
```

导出后 `out-t320/` 里应有 `vision.onnx`、`embed.onnx`、`decoder.onnx` 及对应 `.data`、`meta.json`。
eager 对照官方 `reference_logits` 的 max|Δ| 大约 1e-2 量级（bf16），大于 2e-2 要停下来查。

## 流程 2：ONNX → TensorRT FP16（4090）

**不要转 embed。** 只转 vision + decoder。

```bash
CUDA_VISIBLE_DEVICES=1 python to_trt.py \
    --src /media/workspace/zhouhe/decider-onnx/out-t320 \
    --dst /media/workspace/zhouhe/decider-onnx/out-trt
```

产物：

- `out-trt/vision_fp16.engine`（约 0.7GB，动态 N profile 64/400/576）
- `out-trt/decoder_fp16.engine`（约 6.9GB，静态 T=320）
- `out-trt/trt_meta.json`

`--int8` 是随机数校准，vision 实测层没真正量化，和 FP16 bit-exact；decoder 含 Scan，不要指望 TRT INT8。需要小体积时用下面的 ORT 权重量化，不是 TRT INT8。

引擎带 `HardwareCompatibilityLevel.AMPERE_PLUS`，3070 理论上能 load，但 8GB 显存放不下 decoder FP16 engine。

## 流程 2b：静态 batch=4 / 8（4090 实测）

vision 没有 batch 维，只重导 decoder。单路会按 B 行 padding，**比 B=1 更慢**；只有凑满 B 张图/框一起进 decoder 才有吞吐收益。

```bash
python export_onnx.py --ckpt .../ckpt --out .../out-b8 --only decoder --T 320 --Q 2 --B 8
python to_trt.py --src .../out-b8 --dst .../out-trt --only decoder --tag b8 --workspace-gb 16
python bench_trt.py \
    --engine .../out-trt/decoder_fp16.engine \
    --engine .../out-trt/decoder_b4_fp16.engine \
    --engine .../out-trt/decoder_b8_fp16.engine
```

4090 实测（vision 10.7 ms 逐张 + decoder；dummy 320 输入）：

| | decoder 一次 | decoder/张 | e2e/张 | vs B=1 e2e |
|---|---|---|---|---|
| B=1 | 29.4 ms | 29.4 ms | 40.1 ms | ×1.00 |
| B=4 | 82.7 ms | 20.7 ms | 31.4 ms | ×1.28 |
| B=8 | 165.0 ms | 20.6 ms | 31.3 ms | ×1.28 |

B=8 相对 B=4 没有再快。单路静态 B>1 会 padding，比 B=1 慢。
引擎：`/media/workspace/zhouhe/decider-onnx/out-trt/decoder_b{4,8}_fp16.engine`。

## 流程 3：可选 ORT INT8（3070 / 无 TRT）

这是 ONNX Runtime 的 **weight-only**（`MatMulNBits` / QInt8 Gather），激活仍 fp32。TRT 解析不了 `MatMulNBits`。

```bash
python quantize_int8.py \
    --src /media/workspace/zhouhe/decider-onnx/out-t320 \
    --dst /media/workspace/zhouhe/decider-onnx/out-int8
```

3070 8GB 用这套：`vision.onnx` + `embed.onnx` + `decoder.onnx`（T=320）。

若手里只有旧的 T=256 INT8 decoder、不想重导 3GB 权重，可用 `patch_decoder_t.py` 把图上的 T 改成 320（不改权重）。优先还是重新 `export_onnx.py --T 320`。

## 跑 demo

`demo.py` 里改三份模型路径和 `ckpt_path`（tokenizer、`config.json`、`decision_config.json` 即可，不需要权重；temperature / codes / image_token_id 从这里读，不再读 `meta.json`）。

4090 TRT：

```python
DECIDER = "/media/workspace/zhouhe/decider-onnx"
vision_model_path  = f"{DECIDER}/out-trt/vision_fp16.engine"
embed_model_path   = f"{DECIDER}/out-int8/embed.onnx"      # 仍 CPU
decoder_model_path = f"{DECIDER}/out-trt/decoder_fp16.engine"
input_size = [3, 320, 320]
gpu_mem_limit_gb = 12.0   # TRT 路径几乎不用这项；ORT 才吃
```

3070 ORT INT8：

```python
vision_model_path  = f"{DECIDER}/out-int8/vision.onnx"
embed_model_path   = f"{DECIDER}/out-int8/embed.onnx"
decoder_model_path = f"{DECIDER}/out-int8/decoder.onnx"    # 或 decoder_t320.onnx
gpu_mem_limit_gb = 6.0
```

一阶段：`stream += [decider, ...]`，`questions["default"]`。
二阶段：YOLO 检框 + `secondary_class_names=["person"]` + `questions["person"]`，**必须把 decider 加进 pipeline**，Insight 才能在框上显示分类。

## 本目录脚本

| 脚本 | 用途 |
|---|---|
| `qwen35_onnx.py` | 导出图：Vision / Embed / Decoder（Scan CHUNK=64） |
| `export_onnx.py` | HF ckpt → 三张 ONNX，默认 T=320 Q=2；`--B 4` 冻静态 batch |
| `to_trt.py` | vision + decoder → TensorRT FP16；`--tag b4` 避免覆盖 B=1 |
| `bench_trt.py` | decoder TRT B=1×4 vs B=4 延迟 |
| `quantize_int8.py` | ORT weight-only INT8 |
| `parity_ort.py` | ONNX Runtime vs 官方 PyTorch logit |
| `patch_decoder_t.py` | 已有 T=256 图改 T=320，不重导权重 |
| `export_decoder_batch.py` | 实验：动态 batch、Q=1；当前节点未用 |
| `demo.py` | CPipe 接入示例 |
