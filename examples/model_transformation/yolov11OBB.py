from ultralytics import YOLO
model = YOLO("/mnt/d/demo_e44798a6e52f44c1951878d27f7f5d09_best.pt")  # load a pretrained model (recommended for training)
path = model.export(format="onnx", dynamic=True)  # export the model to ONNX format
print(path)