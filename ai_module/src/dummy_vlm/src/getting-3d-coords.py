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
