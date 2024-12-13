# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.engine",task="detect")

# Run inference
result = model.predict("https://ultralytics.com/images/bus.jpg")


