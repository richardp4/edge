# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.engine")

# Run inference
results = model.val(data="coco.yaml", batch=1, imgsz=640,verbose=False, device="cuda")


