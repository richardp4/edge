# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Convert the YOLOv8 model to TensorRT format
print("Converting YOLOv8 model to TensorRT format...")

model.export(
    format="engine",      # TensorRT format
    dynamic=False,        # Disable dynamic batch size
    imgsz=(640, 640),     # Input image size
    batch=1,              # Fixed batch size
    workspace=4,          # TensorRT workspace size in GB
    simplify=True,        # Simplify the ONNX model
    data="coco128.yaml"   # Dataset configuration
)

