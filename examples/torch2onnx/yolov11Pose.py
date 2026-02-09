from ultralytics import YOLO
model = YOLO("/mnt/d/yolo11x-pose.pt")  # load a pretrained model (recommended for training)
path = model.export(format="onnx", dynamic=True, nms=True, batch=16)  # export the model to ONNX format
# path = model.export(format="engine", dynamic=True, nms=True, batch=16)  # export the model to ONNX format
print(path)