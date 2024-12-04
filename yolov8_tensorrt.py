# -*- coding: utf-8 -*-
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to TensorRT format with specified settings
model.export(
    format="engine",
    dynamic=True,
    batch=8,
    workspace=4,
    data="coco128.yaml"
)

# Load the exported TensorRT model
tensorrt_model = YOLO("yolov8n.engine")

# Perform inference on a local image
results = tensorrt_model("test.jpg")  # 로컬 이미지 경로 사용

# Process and display results
for result in results:
    print("Bounding boxes:", result.boxes)  # 박스 정보 출력
    print("Class probabilities:", result.probs)  # 클래스 확률 출력
    print("Keypoints:", result.keypoints)  # 키포인트 출력 (있을 경우)
    print("Oriented Bounding Boxes:", result.obb)  # OBB 출력 (있을 경우)
    
    # Save the annotated result to an image
    result.save("output_image.jpg")  # 결과 이미지 저장
