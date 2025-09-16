#!/usr/bin/env python3
"""
Fetch an image from an HTTP(S) endpoint every N seconds, save it to the
repo's default data/image.jpg path used by object_detection.py, and then
run the detector.

Usage:
  ./fetch_and_detect.py --url http://localhost:8000/frame.jpg --interval 10 --model yolov8n.pt

Notes:
  - Uses urllib from the standard library (no extra deps).
  - Requires Python 3 and ultralytics installed (for object_detection.py).
"""
import argparse
import os
import sys
import time
import urllib.request
import urllib.error
import shutil
import subprocess
import tempfile


def resolve_default_image_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "image.jpg")


def download_image(url: str, dest_path: str, timeout: float = 10.0):
    """Download image bytes from URL and atomically move to dest_path.

    Returns (ok: bool, info: str) where info is content-type or error.
    """
    tmp_path = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            fd, tmp_path = tempfile.mkstemp(prefix="img_", suffix=".tmp")
            with os.fdopen(fd, "wb") as tmp_file:
                shutil.copyfileobj(resp, tmp_file)
        shutil.move(tmp_path, dest_path)
        return True, content_type
    except Exception as e:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False, str(e)


def run_detection(image_path: str, model: str = "yolov8n.pt") -> int:
    """Invoke object_detection.py on the saved image and return its exit code."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detector = os.path.join(script_dir, "object_detection.py")
    if not os.path.isfile(detector):
        print("Error: object_detection.py not found next to this script", file=sys.stderr)
        return 127

    py = sys.executable or "python3"
    cmd = [py, detector, image_path, "--model", model]
    try:
        # Stream output directly to this process' stdout/stderr
        proc = subprocess.run(cmd)
        return proc.returncode
    except FileNotFoundError:
        print("Error: Python interpreter not found. Ensure python3 is installed.", file=sys.stderr)
        return 127


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch image from endpoint every N seconds and run YOLO detection."
    )
    parser.add_argument("--url", required=True, help="HTTP(S) endpoint returning an image")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between fetches (default: 10)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model name/path (default: yolov8n.pt)")
    parser.add_argument("--save", help="Optional override save path (default: <repo_root>/data/image.jpg)")
    parser.add_argument("--once", action="store_true", help="Fetch and detect once, then exit")
    args = parser.parse_args()

    image_path = args.save or resolve_default_image_path()
    print(f"Saving images to: {image_path}")

    try:
        while True:
            ok, info = download_image(args.url, image_path)
            if ok:
                print(f"[INFO] Downloaded image (Content-Type: {info}) -> {image_path}")
                rc = run_detection(image_path, args.model)
                print(f"[INFO] Detection exit code: {rc}")
            else:
                print(f"[WARN] Failed to download image: {info}", file=sys.stderr)

            if args.once:
                break
            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        print("Interrupted; exiting.")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
