"""Export the Ultralytics YOLOv8 checkpoint to ONNX for the YOLOv8 node."""
from pathlib import Path

from ultralytics import YOLO


def main():
    model = YOLO("model_files/QR_yolov8.pt")
    model.export(format="onnx", imgsz=640, simplify=False, opset=12, dynamic=True)

if __name__ == "__main__":
    main()
