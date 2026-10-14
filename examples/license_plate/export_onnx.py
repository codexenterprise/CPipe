"""Export the PP-Vehicle PP-OCRv3 plate models to ONNX.

Download the inference models first:

    mkdir -p model_files/paddleplate
    cd model_files/paddleplate
    curl -L -o ch_PP-OCRv3_det_infer.tar.gz https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz
    curl -L -o ch_PP-OCRv3_rec_infer.tar.gz https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz
    tar -xzf ch_PP-OCRv3_det_infer.tar.gz
    tar -xzf ch_PP-OCRv3_rec_infer.tar.gz

Then run this script from the repository root.

Use PaddlePaddle 2.6 and paddle2onnx 1.2. PaddlePaddle 3 converts these
inference models through PIR, and paddle2onnx 2.x fails on that graph.
hard_swish raises the ONNX opset from 11 to 14 during export.
"""

import os
import subprocess


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(ROOT, "model_files", "paddleplate")

EXPORTS = [
    ("ch_PP-OCRv3_det_infer", "ch_PP-OCRv3_det.onnx"),
    ("ch_PP-OCRv3_rec_infer", "ch_PP-OCRv3_rec.onnx"),
]


def export_one(model_dir, save_file):
    command = [
        "paddle2onnx",
        "--model_dir", model_dir,
        "--model_filename", "inference.pdmodel",
        "--params_filename", "inference.pdiparams",
        "--save_file", save_file,
        "--opset_version", "14",
        "--enable_onnx_checker", "True",
    ]
    subprocess.run(command, check=True)


def main():
    for folder, filename in EXPORTS:
        model_dir = os.path.join(MODEL_DIR, folder)
        save_file = os.path.join(MODEL_DIR, filename)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"missing inference model directory: {model_dir}")
        export_one(model_dir, save_file)
        print(save_file)


if __name__ == "__main__":
    main()
