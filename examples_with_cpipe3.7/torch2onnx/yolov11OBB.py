from ultralytics import YOLO
model = YOLO("/mnt/d/best.pt")  # load a pretrained model (recommended for training)
path = model.export(format="onnx", dynamic=True)  # export the model to ONNX format
print(path)