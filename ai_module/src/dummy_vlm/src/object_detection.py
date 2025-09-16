#!/usr/bin/env python3
from ultralytics import YOLO
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on an image and print centers.")
    parser.add_argument("image", nargs="?", help="Path to input image. Default: <repo_root>/data/image.jpg")
    parser.add_argument("--model", default="yolov8n.pt", help="Path or name of YOLOv8 model (default: yolov8n.pt)")
    args = parser.parse_args()

    # Resolve default image under repo root if not provided
    if not args.image:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        args.image = os.path.join(project_root, "data", "image.jpg")

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 2

    # Load model
    model = YOLO(args.model)

    # Inference
    results = model(args.image)
    if not results:
        print("No results returned by the model.")
        return 1

    result = results[0]

    # Handle no detections
    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
        return 0

    # Loop through detected objects
    names = getattr(result, "names", {}) or {}
    for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
        x1, y1, x2, y2 = [float(v) for v in box]  # bounding box coordinates
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        label = names.get(int(cls_id), str(int(cls_id)))
        print(f"Detected {label} at center ({x_center:.1f}, {y_center:.1f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





