# -*- coding: utf-8 -*-
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the YOLOv8 model to TensorRT format
model.export(
    format="engine",        # Export to TensorRT format
    dynamic=True,           # Enable dynamic shape
    batch=8,                # Specify the batch size
    workspace=4,            # Set workspace size (in GB)
    data="coco128.yaml"     # Path to dataset YAML for export
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

# Process and display results
for result in results:
    # Optional: Extract individual outputs
    boxes = result.boxes  # Bounding boxes
    masks = result.masks  # Segmentation masks (if available)
    keypoints = result.keypoints  # Keypoints (if available)
    probs = result.probs  # Classification probabilities
    obb = result.obb  # Oriented bounding boxes (if available)
    
    # Display the annotated image on the screen
    result.show()  # Show result in a window
