from ultralytics import YOLO


from ultralytics import YOLO

# Load a YOLO26 model
# model = YOLO("yolo26n.pt")

# Export the model to RKNN format
# model.export(format="rknn", name="rk3588")  # creates '/yolo26n_rknn_model'

# Export an INT8-quantized RKNN model with calibration data


# Load the YOLO11 model
# model = YOLO("model_files/yolo26n.pt")

# # Export the model to ONNX format opt 19
# model.export(format="onnx", dynamic=True, imgsz=(640, 640), opset=19)
# model.export(format="rknn", imgsz=(640, 640), name="rk3588")


# Load the exported RKNN model
model = YOLO("./model_files/yolo26n_rknn_model")

# Run inference
results = model("https://ultralytics.com/images/bus.jpg")
print(results)
# Load the exported ONNX model
# onnx_model = YOLO("model_files/yolo26n.onnx", task="detect")
# print(onnx_model)

# Onnx 转 TensorRT
# /home/mafneg/TensorRT-8.5.3.1/bin/trtexec --onnx=yolo11s-seg.onnx  --saveEngine=yolo11s-seg.engine --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640  --maxShapes=images:32x3x640x640   --fp16