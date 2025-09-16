import math
import numpy as np
import cv2
import os
from collections import defaultdict

# Dedup threshold (in same units as computed positions). Default 100; override via env DEDUP_DISTANCE
DEDUP_DISTANCE = float(os.getenv("DEDUP_DISTANCE", "100.0"))

def equirect_to_direction(u, v, W, H):
    print("doing equirect to direction")
    yaw = 2.0 * math.pi * (u / W - 0.5)
    pitch = math.pi * (0.5 - v / H)
    x = math.cos(pitch) * math.cos(yaw)
    y = math.cos(pitch) * math.sin(yaw)
    z = math.sin(pitch)
    return np.array([x, y, z])

def object_position_3d(u, v, distance, current_position, image_path="../data/image.png"):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    direction = equirect_to_direction(u, v, W, H)
    pos_obj = np.array(current_position) + direction * distance
    return pos_obj

input_file = "../data/object_list.txt"
output_file = "../data/object_list_updated.txt"
image_path = "../data/image.png"

if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

kept_by_label = defaultdict(list)  # label -> list of np.array([x,y,z]) for kept entries
skipped_duplicates = 0
total_in = 0
total_out = 0

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        total_in += 1
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 7:
            print(f"Skipping malformed line: {line}")
            continue
        
        timestamp = parts[0]
        label = parts[1]
        u = float(parts[2])
        v = float(parts[3])
        distance = float(parts[4])
        current_position = [float(parts[5]), float(parts[6]), 0.0]  # z of robot assumed 0 if not given
        
        # Skip objects with invalid distance
        if distance == -1:
            continue
        
        pos_3d = object_position_3d(u, v, distance, current_position, image_path)
        # Check per-class deduplication: skip if any kept position for this label is within threshold
        is_dup = any(np.linalg.norm(pos_3d - p) <= DEDUP_DISTANCE for p in kept_by_label[label])
        if is_dup:
            skipped_duplicates += 1
            # Optional: uncomment for verbose logs
            # print(f"Skipping duplicate {label} within {DEDUP_DISTANCE} of an existing entry")
            continue

        kept_by_label[label].append(pos_3d)
        f_out.write(f"{timestamp} {label} {pos_3d[0]:.3f} {pos_3d[1]:.3f} {pos_3d[2]:.3f}\n")
        total_out += 1

print(f"Updated object list saved to {output_file}")
print(f"Input lines: {total_in}, Written: {total_out}, Skipped as duplicates: {skipped_duplicates}, Threshold: {DEDUP_DISTANCE}")
