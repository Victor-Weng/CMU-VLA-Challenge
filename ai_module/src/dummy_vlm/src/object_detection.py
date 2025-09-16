from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Run detection on a single image
results = model("/home/hector/Downloads/roboflow/image.jpg")
result = results[0]  # get the first (and only) result

# Loop through detected objects
for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
    x1, y1, x2, y2 = box  # bounding box coordinates
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    label = result.names[int(cls_id)]  # convert class ID to label

    print(f"Detected {label} at center ({x_center:.1f}, {y_center:.1f})")
