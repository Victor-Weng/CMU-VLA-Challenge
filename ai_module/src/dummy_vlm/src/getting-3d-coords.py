import math
import numpy as np
import cv2
import os

def equirect_to_direction(u, v, W, H):
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

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
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
        f_out.write(f"{timestamp} {label} {pos_3d[0]:.3f} {pos_3d[1]:.3f} {pos_3d[2]:.3f}\n")

print(f"Updated object list saved to {output_file}")
