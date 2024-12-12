# -*- coding: utf-8 -*-
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
results = tensorrt_model(image_path)

# Process and display the results
for result in results:
    print("Bounding boxes:", result.boxes.xyxy)  # Bounding box coordinates
    print("Class probabilities:", result.probs)  # Class probabilities
    if result.keypoints is not None:            # Check for keypoints
        print("Keypoints:", result.keypoints)
    if result.obb is not None:                  # Check for oriented bounding boxes
        print("Oriented Bounding Boxes:", result.obb)
    
    # Save the annotated result image
    result.save("output_image.jpg")  # Save the result image
