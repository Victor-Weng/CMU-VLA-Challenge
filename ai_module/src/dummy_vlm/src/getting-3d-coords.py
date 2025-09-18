import math
import numpy as np
import cv2
import os
import sys
from collections import defaultdict

# Dedup threshold (in same units as computed positions). Default 30; override via env DEDUP_DISTANCE
DEDUP_DISTANCE = float(os.getenv("DEDUP_DISTANCE", "30.0"))

# Resolve package paths relative to this file regardless of current working directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))  # dummy_vlm/
DATA_DIR = os.path.join(PKG_DIR, 'data')
# Common alternate location used by the saver node (ai_module/src/data)
ALT_DATA_DIR = os.path.abspath(os.path.join(PKG_DIR, '..', '..', 'data'))
ENV_IMAGE_DIR = os.getenv('DETECT_IMAGE_DIR', '').strip()

def equirect_to_direction(u, v, W, H):
    # print("doing equirect to direction")
    yaw = 2.0 * math.pi * (u / W - 0.5)
    pitch = math.pi * (0.5 - v / H)
    x = math.cos(pitch) * math.cos(yaw)
    y = math.cos(pitch) * math.sin(yaw)
    z = math.sin(pitch)
    return np.array([x, y, z])

_warned_missing_image = False
_cached_shape = None  # (H, W)

def _resolve_image_shape(candidates):
    """Return (H, W) from the first readable image among candidates or use env/defaults."""
    global _warned_missing_image, _cached_shape
    if _cached_shape is not None:
        return _cached_shape
    # Try provided candidates
    for p in candidates:
        if p and os.path.isfile(p):
            img = cv2.imread(p)
            if img is not None:
                _cached_shape = img.shape[:2]
                return _cached_shape
    # Try scanning likely dirs for any PNG/JPG to infer shape
    for d in [ENV_IMAGE_DIR, DATA_DIR, ALT_DATA_DIR]:
        if not d:
            continue
        try:
            for name in os.listdir(d):
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(d, name))
                    if img is not None:
                        _cached_shape = img.shape[:2]
                        return _cached_shape
        except Exception:
            pass
    # Env override
    try:
        w_env = int(os.getenv('IMAGE_WIDTH', '0'))
        h_env = int(os.getenv('IMAGE_HEIGHT', '0'))
        if w_env > 0 and h_env > 0:
            _cached_shape = (h_env, w_env)
            return _cached_shape
    except Exception:
        pass
    # Default fallback
    if not _warned_missing_image:
        print("Warning: could not determine image size; using default 1024x512. Set IMAGE_WIDTH/IMAGE_HEIGHT to override.")
        _warned_missing_image = True
    _cached_shape = (512, 1024)
    return _cached_shape

def object_position_3d(u, v, distance, current_position, image_candidates):
    H, W = _resolve_image_shape(image_candidates)
    direction = equirect_to_direction(u, v, W, H)
    pos_obj = np.array(current_position) + direction * distance
    return pos_obj

input_file = os.path.join(DATA_DIR, "object_list_detect.txt")
if not os.path.isfile(input_file):
    # Fallback to legacy filename if detect log not present
    legacy = os.path.join(DATA_DIR, "object_list.txt")
    input_file = legacy

output_file = os.path.join(DATA_DIR, "object_list_updated.txt")
image_path = os.path.join(DATA_DIR, "image.png")

if not os.path.isfile(input_file):
    msg = f"Input file not found: {input_file}. Expected either object_list_detect.txt or object_list.txt in {DATA_DIR}"
    print(msg)
    # Ensure expected output file exists, even if empty, so downstream readers don't fail to open
    os.makedirs(DATA_DIR, exist_ok=True)
    open(output_file, "w").close()
    sys.exit(0)

kept_by_label = defaultdict(list)  # label -> list of np.array([x,y,z]) for kept entries
skipped_duplicates = 0
total_in = 0
total_out = 0

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        total_in += 1
        if not line.strip():
            continue
        # Lines are tab-separated: ts, label, u, v, dist, robot_x, robot_y, image_name
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 8:
            print(f"Skipping malformed line (need 8 tab-separated columns): {line.strip()}")
            continue
        timestamp = parts[0]
        label = parts[1]
        try:
            u = float(parts[2])
            v = float(parts[3])
            distance = float(parts[4])
            current_position = [float(parts[5]), float(parts[6]), 0.0]
        except ValueError:
            print(f"Skipping line with invalid numeric fields: {line.strip()}")
            continue
        image_name = parts[7]
        
        # Skip objects with invalid distance
        if distance == -1:
            continue
        
        # Resolve the per-line image path across likely directories
        candidates = []
        if image_name:
            if ENV_IMAGE_DIR:
                candidates.append(os.path.join(ENV_IMAGE_DIR, image_name))
            candidates.append(os.path.join(DATA_DIR, image_name))
            candidates.append(os.path.join(ALT_DATA_DIR, image_name))
        else:
            # Fall back to default image path
            candidates.append(image_path)
        # Compute 3D position using resolved or fallback image shape
        pos_3d = object_position_3d(u, v, distance, current_position, candidates)
        # Check per-class deduplication: skip if any kept position for this label is within threshold
        is_dup = any(np.linalg.norm(pos_3d - p) <= DEDUP_DISTANCE for p in kept_by_label[label])
        if is_dup:
            skipped_duplicates += 1
            # Optional: uncomment for verbose logs
            # print(f"Skipping duplicate {label} within {DEDUP_DISTANCE} of an existing entry")
            continue

        kept_by_label[label].append(pos_3d)
        f_out.write(f"{label} {pos_3d[0]:.3f} {pos_3d[1]:.3f} {pos_3d[2]:.3f}\n")
        total_out += 1

print(f"Updated object list saved to {output_file}")
print(f"Input lines: {total_in}, Written: {total_out}, Skipped as duplicates: {skipped_duplicates}, Threshold: {DEDUP_DISTANCE}")
