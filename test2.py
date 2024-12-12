# -*- coding: utf-8 -*-
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the YOLOv8 model to TensorRT format
model.export(
    format="engine",
    dynamic=True,
    batch=8,
    workspace=4,
    data="coco128.yaml"
)

# Load the exported TensorRT model
tensorrt_model = YOLO("yolov8n.engine")

# Perform inference on a test image
image_path = "test.jpg"  # Local image path

# Measure inference time
start_time = time.time()
results = tensorrt_model(image_path)  # Run inference
end_time = time.time()

# Calculate and print inference time
inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")
