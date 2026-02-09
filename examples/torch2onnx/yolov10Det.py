from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/mnt/d/yolov10x.pt")

# Export the model to ONNX format
model.export(format="onnx", dynamic=True, imgsz=(640, 640))

# Load the exported ONNX model
onnx_model = YOLO("yolov10x.onnx")

# Onnx 转 TensorRT
# /home/mafneg/TensorRT-8.5.3.1/bin/trtexec --onnx=yolo11s-seg.onnx  --saveEngine=yolo11s-seg.engine --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640  --maxShapes=images:32x3x640x640   --fp16