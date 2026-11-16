#!/usr/bin/env python3
"""把 Decider 的 FP16 ONNX 编成 TensorRT engine（在 4090 上跑）。

只转 vision + decoder。embed 是 CPU Gather，转 TRT 更慢还占显存，cpipe 节点也不会加载 embed.engine。

  CUDA_VISIBLE_DEVICES=1 python to_trt.py \\
      --src /path/out-t320 --dst /path/out-trt

产物：
  vision_fp16.engine   动态 N，profile 64/400/576（对应约 256/320/384 边长）
  decoder_fp16.engine  静态图，要求 ONNX 已冻 T（推荐 320）

--int8 用随机数校准，vision 实测没真正量化（和 FP16 一样）；decoder 含 Scan，不要指望 INT8。
小体积用 quantize_int8.py（ORT MatMulNBits），不是这条路径。

需要 tensorrt Python 包。若 import tensorrt 失败，把 site-packages/tensorrt_libs 加进 LD_LIBRARY_PATH。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tensorrt as trt

# 320 letterbox → 20x20 patch = 400；384 → 576；256 → 256。min 留一点余量。
VISION_N = {"min": 64, "opt": 400, "max": 576}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TorchCalibrator(trt.IInt8EntropyCalibrator2):
    """随机数校准器，只为让 INT8 build 能走完。不能当量化质量保证。"""

    def __init__(self, shapes: dict[str, tuple], dtypes: dict[str, str], cache: Path, n=8):
        super().__init__()
        import torch
        self.torch = torch
        self.shapes = list(shapes.items())
        self.dtypes = dtypes
        self.cache = cache
        self.n = n
        self.i = 0
        self._keep = []

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.i >= self.n:
            return None
        self.i += 1
        self._keep = []
        ptrs = []
        torch = self.torch
        for name, shape in self.shapes:
            if name in ("interp_indices", "rope_pos_ids", "lengths", "markers", "position_ids"):
                t = torch.zeros(shape, device="cuda", dtype=torch.int64)
                if name == "lengths":
                    t = torch.tensor([180], device="cuda", dtype=torch.int64)
                elif name == "markers":
                    t = torch.tensor([[40, 80]], device="cuda", dtype=torch.int64)
                elif name == "position_ids":
                    t = torch.arange(shape[-1], device="cuda", dtype=torch.int64).view(1, 1, -1).expand(shape).contiguous()
            else:
                t = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.02
            self._keep.append(t)
            ptrs.append(int(t.data_ptr()))
        return ptrs

    def read_calibration_cache(self):
        if self.cache.exists():
            return self.cache.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache.write_bytes(bytes(cache))


def build(onnx_path: Path, engine_path: Path, profiles: dict | None, *, fp16: bool, int8: bool, workspace_gb: int, hw_compat: bool):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    log(f"parse {onnx_path}")
    ok = parser.parse_from_file(str(onnx_path))
    if not ok:
        for i in range(parser.num_errors):
            log(f"  parse error: {parser.get_error(i)}")
        return False
    log(f"  layers={network.num_layers} inputs={network.num_inputs} outputs={network.num_outputs}")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        log(f"  in {t.name} {tuple(t.shape)} {t.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log("flag FP16")
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        log("flag INT8")
    # AMPERE_PLUS：同系列卡（3070 等）理论上能 load。3070 8GB 仍放不下 decoder FP16。
    if hw_compat and hasattr(trt, "HardwareCompatibilityLevel"):
        try:
            config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
            log("hardwareCompatibility AMPERE_PLUS")
        except Exception as e:
            log(f"hw compat skipped: {e}")

    if profiles:
        profile = builder.create_optimization_profile()
        for name, (mn, opt, mx) in profiles.items():
            profile.set_shape(name, mn, opt, mx)
            log(f"  profile {name} min={mn} opt={opt} max={mx}")
        config.add_optimization_profile(profile)
        cal_shapes = {n: tuple(opt) for n, (_, opt, _) in profiles.items()}
    else:
        cal_shapes = {network.get_input(i).name: tuple(int(d) for d in network.get_input(i).shape) for i in range(network.num_inputs)}
    if int8:
        try:
            cal = TorchCalibrator(cal_shapes, {}, engine_path.with_suffix(".calib"))
            config.int8_calibrator = cal
            log(f"INT8 torch calibrator ready {list(cal_shapes)}")
        except Exception as e:
            log(f"INT8 calibrator unavailable ({e}); building without it")

    t0 = time.time()
    log("build ...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        log("BUILD FAILED")
        return False
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(engine))
    log(f"wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description="Decider ONNX → TensorRT (vision + decoder, no embed)")
    ap.add_argument("--src", required=True, help="FP16 ONNX 目录，需有 vision.onnx 和 decoder.onnx")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--only", default="vision,decoder", help="不要加 embed")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--int8", action="store_true", help="随机校准，质量无保证；decoder 不建议")
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--no-hw-compat", action="store_true")
    ap.add_argument("--vision-n", default="64,400,576", help="min,opt,max num_patches")
    ap.add_argument("--decoder-name", default="decoder.onnx", help="静态 T 的 decoder 文件名")
    ap.add_argument("--tag", default="", help="engine 文件名插入段，如 b4 → decoder_b4_fp16.engine")
    args = ap.parse_args()
    fp16 = not args.no_fp16
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if "embed" in only:
        log("embed 不转 TRT，已忽略。推理用 CPU ONNX。")
        only.discard("embed")
    nmin, nopt, nmax = (int(x) for x in args.vision_n.split(","))
    results = {}

    if "vision" in only:
        vis = src / "vision.onnx"
        # 1536 = patch embed 输入宽（官方 visual hidden）；N 是动态 patch 数。
        profiles = {
            "pixel_values": ((nmin, 1536), (nopt, 1536), (nmax, 1536)),
            "interp_indices": ((nmin, 4), (nopt, 4), (nmax, 4)),
            "interp_weights": ((nmin, 4), (nopt, 4), (nmax, 4)),
            "rope_pos_ids": ((nmin, 2), (nopt, 2), (nmax, 2)),
        }
        prec = "int8" if args.int8 else "fp16"
        vname = f"vision_{args.tag}_{prec}.engine" if args.tag else f"vision_{prec}.engine"
        results["vision"] = build(
            vis, dst / vname, profiles,
            fp16=fp16, int8=args.int8, workspace_gb=args.workspace_gb, hw_compat=not args.no_hw_compat,
        )

    if "decoder" in only:
        dec = src / args.decoder_name
        # 静态 T 图不需要 profile。若 ONNX 仍是动态 T，TRT 会在 parse/build 失败。
        prec = "int8" if args.int8 else "fp16"
        dname = f"decoder_{args.tag}_{prec}.engine" if args.tag else f"decoder_{prec}.engine"
        results["decoder"] = build(
            dec, dst / dname, None,
            fp16=fp16, int8=args.int8, workspace_gb=args.workspace_gb, hw_compat=not args.no_hw_compat,
        )

    (dst / "trt_meta.json").write_text(json.dumps({
        "src": str(src), "fp16": fp16, "int8": args.int8,
        "hardware_compat": not args.no_hw_compat, "results": results,
        "trt": trt.__version__,
        "note": "embed stays CPU ONNX; INT8 here is dummy calibration",
    }, indent=2), encoding="utf-8")
    failed = [k for k, v in results.items() if not v]
    if failed:
        log(f"FAILED: {failed}")
        sys.exit(1)
    log("all ok")


if __name__ == "__main__":
    main()
