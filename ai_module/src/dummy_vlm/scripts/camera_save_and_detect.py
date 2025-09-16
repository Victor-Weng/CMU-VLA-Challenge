#!/usr/bin/env python3
import os
import sys
import subprocess
import errno

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
try:
    from cv_bridge import CvBridge, CvBridgeError
except Exception:
    CvBridge = None
    class CvBridgeError(Exception):
        pass
import cv2
import numpy as np


class CameraSaveAndDetect:
    @staticmethod
    def _rosimg_to_cv2(msg):
        enc = (getattr(msg, 'encoding', '') or '').lower()
        width = msg.width
        height = msg.height
        step = msg.step
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        # mono8 path
        if enc in ('mono8', '8uc1'):
            if step == width:
                img = buf.reshape((height, width))
            else:
                img = buf.reshape((height, step))[:, :width]
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # assume 3 channels
        channels = 3
        if step == width * channels:
            img = buf.reshape((height, width, channels))
        else:
            img = buf.reshape((height, step))[:, :width * channels].reshape((height, width, channels))
        if enc in ('rgb8', 'rgb16'):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if enc in ('rgba8', 'bgra8'):
            if enc.startswith('rgba'):
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # default treat as BGR
        return img
    def __init__(self):
        self.image_topic = rospy.get_param('~image_topic', '/camera/image')
        self.interval_sec = float(rospy.get_param('~interval_sec', 10.0))
        self.model = rospy.get_param('~model', 'yolov8n.pt')
        self.save_path_param = rospy.get_param('~save_path', '')  # if provided, overrides default path
        self.extra_compat_writes = bool(rospy.get_param('~extra_compat_writes', True))
        # Gating: wait for a question before doing anything
        self.wait_for_question = bool(rospy.get_param('~wait_for_question', True))
        self.question_topic = rospy.get_param('~question_topic', '/challenge_question')
        self.activated = not self.wait_for_question
        # New: sequential saving and cleanup
        self.sequential_saves = bool(rospy.get_param('~sequential_saves', True))
        self.image_dir_param = rospy.get_param('~image_dir', '')
        self.image_prefix = rospy.get_param('~image_prefix', 'image_')
        self.cleanup_images_on_startup = bool(rospy.get_param('~cleanup_images_on_startup', True))
    # Shutdown when final answer is published
    self.shutdown_on_final_answer = bool(rospy.get_param('~shutdown_on_final_answer', True))

        # Compute save path to match object_detection.py's default
        # object_detection.py lives in <pkg_dir>/src/object_detection.py
        this_dir = os.path.dirname(os.path.abspath(__file__))  # <pkg_dir>/scripts
        pkg_dir = os.path.abspath(os.path.join(this_dir, '..'))
        objdet_dir = os.path.join(pkg_dir, 'src')
        project_root = os.path.abspath(os.path.join(objdet_dir, '..', '..'))  # mimics object_detection.py logic
        data_dir = os.path.join(project_root, 'data')
        default_save_path = os.path.join(data_dir, 'image.png')
        self.save_path = self.save_path_param if self.save_path_param else default_save_path
        # Resolve sequential image directory and extension
        self.image_ext = os.path.splitext(self.save_path)[1].lower() or '.png'
        if self.image_dir_param:
            self.image_dir = self.image_dir_param
        else:
            # Use directory of save_path as default
            self.image_dir = os.path.dirname(self.save_path)
        self.detector_script = os.path.join(objdet_dir, 'object_detection.py')

        # Also compute alternate save paths for compatibility, if enabled
        alt_root = os.path.abspath(os.path.join(objdet_dir, '..', '..', '..'))  # -> <repo>/ai_module
        self.alt_data_dir = os.path.join(alt_root, 'data')  # <repo>/ai_module/data
        self.alt_save_path = os.path.join(self.alt_data_dir, os.path.basename(self.save_path) or 'image.png')
        # A second alternative matching <repo>/ai_module/src/data (as seen in earlier logs)
        alt_src_data_dir = os.path.join(alt_root, 'src', 'data')
        self.alt2_data_dir = alt_src_data_dir
        self.alt2_save_path = os.path.join(alt_src_data_dir, os.path.basename(self.save_path) or 'image.png')

        # Ensure output dirs exist
        try:
            os.makedirs(self.image_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.bridge = CvBridge() if CvBridge is not None else None
        self.last_save_time = rospy.Time(0)
        self.last_rx_time = rospy.Time(0)

        # Optional cleanup on startup
        if self.cleanup_images_on_startup and self.sequential_saves:
            self._cleanup_images()
        # Initialize sequential counter
        self._init_seq_counter()

        # Optional question subscriber to activate on first question
        if self.wait_for_question:
            self.q_sub = rospy.Subscriber(self.question_topic, String, self._on_question, queue_size=5)
            rospy.loginfo('camera_save_and_detect: waiting for first question on %s before activating', self.question_topic)
        if self.shutdown_on_final_answer:
            self.final_sub = rospy.Subscriber('/final_answer', String, self._on_final_answer, queue_size=1)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        rospy.loginfo('camera_save_and_detect: listening to %s', self.image_topic)
        rospy.loginfo('camera_save_and_detect: interval = %.2fs, model = %s', self.interval_sec, self.model)
        if self.sequential_saves:
            rospy.loginfo('camera_save_and_detect: sequential saving to dir=%s prefix=%s ext=%s (start index=%d)', self.image_dir, self.image_prefix, self.image_ext, getattr(self, 'seq_index', 1))
        else:
            rospy.loginfo('camera_save_and_detect: primary save path = %s', self.save_path)
        if self.extra_compat_writes:
            rospy.loginfo('camera_save_and_detect: compat save paths = %s, %s', self.alt_save_path, self.alt2_save_path)

    def _on_final_answer(self, _msg: String):
        try:
            rospy.loginfo('camera_save_and_detect: received /final_answer; shutting down this node')
            self.activated = False
            rospy.signal_shutdown('final answer received')
        except Exception:
            pass

    def image_cb(self, msg: Image):
        if not self.activated:
            rospy.loginfo_throttle(10.0, 'camera_save_and_detect: idle (waiting for question on %s)', self.question_topic)
            return
        now = msg.header.stamp if msg.header.stamp and msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        self.last_rx_time = now
        # Basic frame info for debugging
        try:
            enc = msg.encoding
            rospy.logdebug('camera_save_and_detect: received image stamp=%.3f encoding=%s size=%dx%d',
                           now.to_sec(), enc, msg.width, msg.height)
            # Also surface occasional INFO to show liveness
            rospy.loginfo_throttle(5.0, 'camera_save_and_detect: receiving images on %s (encoding=%s %dx%d) ...',
                                   self.image_topic, enc, msg.width, msg.height)
        except Exception:
            pass

        if self.last_save_time.to_sec() > 0:
            elapsed = (now - self.last_save_time).to_sec()
            if elapsed < self.interval_sec:
                rospy.logdebug('camera_save_and_detect: skipping save, %.2fs until next capture', self.interval_sec - elapsed)
                return

        cv_img = None
        if self.bridge is not None:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except CvBridgeError as e:
                rospy.logwarn('cv_bridge bgr8 conversion failed (%s); trying passthrough', str(e))
                try:
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    if len(cv_img.shape) == 2:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
                except CvBridgeError as e2:
                    rospy.logwarn('cv_bridge passthrough failed: %s', str(e2))
                    cv_img = None
        if cv_img is None:
            # Fallback: direct NumPy conversion for common encodings
            cv_img = self._rosimg_to_cv2(msg)
            if cv_img is None:
                rospy.logwarn('Failed to convert ROS Image without cv_bridge (encoding=%s)', getattr(msg, 'encoding', ''))
                return

        # Save as JPEG
        # Decide format based on extension
        # Determine output path
        if self.sequential_saves:
            out_path = self._next_image_path()
        else:
            out_path = self.save_path
        _, ext = os.path.splitext(out_path.lower())
        try:
            if ext == '.png':
                ok = cv2.imwrite(out_path, cv_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            else:
                ok = cv2.imwrite(out_path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                rospy.logwarn('Failed to write image to %s', out_path)
                return
            # Best-effort write to alternate locations if enabled
            if self.extra_compat_writes:
                try:
                    os.makedirs(self.alt_data_dir, exist_ok=True)
                    alt_out = os.path.join(self.alt_data_dir, os.path.basename(out_path))
                    cv2.imwrite(alt_out, cv_img)
                except Exception:
                    pass
                try:
                    os.makedirs(self.alt2_data_dir, exist_ok=True)
                    alt2_out = os.path.join(self.alt2_data_dir, os.path.basename(out_path))
                    cv2.imwrite(alt2_out, cv_img)
                except Exception:
                    pass
        except Exception as e:
            rospy.logwarn('Exception writing image: %s', str(e))
            return

        self.last_save_time = now
        if self.extra_compat_writes:
            rospy.loginfo('Saved image to %s (compat dirs mirrored); running detector...', out_path)
        else:
            rospy.loginfo('Saved image to %s; running detector...', out_path)

        # Run detector as a subprocess
        py = sys.executable or 'python3'
        cmd = [py, self.detector_script, out_path, '--model', self.model]
        rospy.loginfo('Running detector command: %s', ' '.join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            # Stream output to ROS log
            for line in proc.stdout:
                rospy.loginfo('[detector] %s', line.strip())
            rc = proc.wait()
            rospy.loginfo('Detector exited with code %d', rc)
        except Exception as e:
            rospy.logwarn('Failed to run detector: %s', str(e))

    # ---------- Helpers (instance methods) ----------
    def _init_seq_counter(self):
        # Determine next index based on existing files
        if not self.sequential_saves:
            self.seq_index = 1
            return
        try:
            files = os.listdir(self.image_dir)
        except Exception:
            files = []
        max_idx = 0
        pref = self.image_prefix
        for name in files:
            if not name.startswith(pref) or not name.endswith(self.image_ext):
                continue
            num = name[len(pref):-len(self.image_ext)] if len(self.image_ext) > 0 else name[len(pref):]
            if num.isdigit():
                max_idx = max(max_idx, int(num))
        self.seq_index = max_idx + 1

    def _next_image_path(self):
        name = f"{self.image_prefix}{self.seq_index:06d}{self.image_ext}"
        self.seq_index += 1
        return os.path.join(self.image_dir, name)

    def _cleanup_images(self):
        try:
            files = os.listdir(self.image_dir)
        except Exception:
            return
        pref = self.image_prefix
        for name in files:
            if name.startswith(pref) and name.endswith(self.image_ext):
                try:
                    os.remove(os.path.join(self.image_dir, name))
                except Exception:
                    pass

    def _on_question(self, msg: String):
        if not self.activated:
            self.activated = True
            rospy.loginfo('camera_save_and_detect: activated by question: "%s"', (msg.data or '')[:80])


def main():
    rospy.init_node('camera_save_and_detect', anonymous=False)
    node = CameraSaveAndDetect()
    rospy.spin()


if __name__ == '__main__':
    main()
def _bytes_to_array(msg, channels):
    # Handle row stride (step) potentially > width*channels
    width = msg.width
    height = msg.height
    step = msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if step == width * channels:
        return buf.reshape((height, width, channels))
    # Otherwise, slice each row
    arr = buf.reshape((height, step))
    arr = arr[:, :width * channels]
    return arr.reshape((height, width, channels))

def _mono_bytes_to_array(msg):
    width = msg.width
    height = msg.height
    step = msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if step == width:
        return buf.reshape((height, width))
    arr = buf.reshape((height, step))
    return arr[:, :width]

def _convert_encoding_to_bgr(img, encoding):
    enc = (encoding or '').lower()
    if enc in ('bgr8', 'bgr16'):
        return img
    if enc in ('rgb8', 'rgb16'):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ('rgba8', 'bgra8'):
        # Convert to BGR by dropping alpha
        if enc.startswith('rgba'):
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    if enc in ('mono8', '8uc1'):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Unknown: return as-is
    return img

def rosimg_to_cv2_bgr(msg):
    enc = (msg.encoding or '').lower()
    if enc in ('mono8', '8uc1'):
        img = _mono_bytes_to_array(msg)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Assume 3-channel by default
    img = _bytes_to_array(msg, 3)
    return _convert_encoding_to_bgr(img, enc)

# Keep module-level helper available (optional), but class now has its own implementation
