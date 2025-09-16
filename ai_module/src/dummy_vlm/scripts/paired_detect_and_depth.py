#!/usr/bin/env python3
import os
import re
import sys
import math
import errno
import subprocess

import rospy
from sensor_msgs.msg import Image, PointCloud2
try:
    from cv_bridge import CvBridge, CvBridgeError
except Exception:
    CvBridge = None
    class CvBridgeError(Exception):
        pass
import sensor_msgs.point_cloud2 as pc2
import message_filters
import tf
import numpy as np
import cv2


DET_LINE_RE = re.compile(r"^Detected\s+(?P<label>.+?)\s+at\s+center\s*\(\s*(?P<x>[-0-9.]+)\s*,\s*(?P<y>[-0-9.]+)\s*\)\s*$")


class PairedDetectAndDepth:
    def __init__(self):
        # Params
        self.image_topic = rospy.get_param('~image_topic', '/paired/image')
        self.points_topic = rospy.get_param('~points_topic', '/paired/points')
        self.model = rospy.get_param('~model', 'yolov8n.pt')
        self.save_path = rospy.get_param('~save_path', '')  # default computed below
        self.object_list_path = rospy.get_param('~object_list_path', '')
        self.angle_window_deg = float(rospy.get_param('~angle_window_deg', 1.5))

        # Resolve default paths
        this_dir = os.path.dirname(os.path.abspath(__file__))  # <pkg>/scripts
        pkg_dir = os.path.abspath(os.path.join(this_dir, '..'))
        objdet_dir = os.path.join(pkg_dir, 'src')
        project_root = os.path.abspath(os.path.join(objdet_dir, '..', '..'))
        data_dir = os.path.join(project_root, 'data')
        default_image_path = os.path.join(data_dir, 'image.png')
        if not self.save_path:
            self.save_path = default_image_path
        # Default object_list to package data dir
        if not self.object_list_path:
            pkg_data_dir = os.path.join(pkg_dir, 'data')
            self.object_list_path = os.path.join(pkg_data_dir, 'object_list.txt')

        # Ensure dirs exist
        for d in [os.path.dirname(self.save_path), os.path.dirname(self.object_list_path)]:
            try:
                os.makedirs(d)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        self.detector_script = os.path.join(objdet_dir, 'object_detection.py')
    self.bridge = CvBridge() if CvBridge is not None else None
        self.tf_listener = tf.TransformListener()

        # Subscribers with synchronization
        img_sub = message_filters.Subscriber(self.image_topic, Image, queue_size=5)
        pc_sub = message_filters.Subscriber(self.points_topic, PointCloud2, queue_size=5)
        sync = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], queue_size=10, slop=0.10)
        sync.registerCallback(self.on_pair)

        rospy.loginfo('paired_detect_and_depth: listening to %s and %s', self.image_topic, self.points_topic)
        rospy.loginfo('paired_detect_and_depth: saving images to %s, detections to %s', self.save_path, self.object_list_path)

    def on_pair(self, img_msg: Image, pc_msg: PointCloud2):
        stamp = img_msg.header.stamp
        img_frame = img_msg.header.frame_id or 'camera'
        pc_frame = pc_msg.header.frame_id or 'map'
        # Convert and save image
        cv_img = None
        if self.bridge is not None:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            except CvBridgeError as e:
                rospy.logwarn('cv_bridge conversion failed: %s', str(e))
                cv_img = None
        if cv_img is None:
            cv_img = rosimg_to_cv2_bgr(img_msg)
            if cv_img is None:
                rospy.logwarn('Failed to convert image without cv_bridge')
                return
        try:
            import cv2
            ok = False
            ext = os.path.splitext(self.save_path.lower())[1]
            if ext == '.png':
                ok = cv2.imwrite(self.save_path, cv_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            else:
                ok = cv2.imwrite(self.save_path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                rospy.logwarn('Failed to write image to %s', self.save_path)
                return
        except Exception as e:
            rospy.logwarn('Exception saving image: %s', str(e))
            return

        # Run detector script on saved image and parse output
        py = sys.executable or 'python3'
        cmd = [py, self.detector_script, self.save_path, '--model', self.model]
        rospy.loginfo('paired_detect_and_depth: running detector: %s', ' '.join(cmd))
        dets = []  # list of (label, u, v)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in proc.stdout:
                line = line.strip()
                rospy.loginfo('[detector] %s', line)
                m = DET_LINE_RE.match(line)
                if m:
                    label = m.group('label')
                    u = float(m.group('x'))
                    v = float(m.group('y'))
                    dets.append((label, u, v))
            rc = proc.wait()
            if rc != 0:
                rospy.logwarn('Detector exited with code %d', rc)
        except Exception as e:
            rospy.logwarn('Failed running detector: %s', str(e))
            return

        if not dets:
            rospy.loginfo('paired_detect_and_depth: no detections to fuse')
            return

        # Prepare TF transform from point cloud frame to camera frame at the image timestamp
        try:
            self.tf_listener.waitForTransform(img_frame, pc_frame, stamp, rospy.Duration(0.2))
            (trans, rot) = self.tf_listener.lookupTransform(img_frame, pc_frame, stamp)
            T = self._tf_to_matrix(trans, rot)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn('TF transform %s -> %s at t=%.3f unavailable: %s', pc_frame, img_frame, stamp.to_sec(), str(e))
            T = None

        # Read points and optionally transform to camera frame
        points = pc2.read_points(pc_msg, field_names=['x', 'y', 'z'], skip_nans=True)
        pts_cam = []
        if T is None:
            # Assume cloud already in camera frame (best-effort)
            for p in points:
                pts_cam.append((p[0], p[1], p[2]))
        else:
            # Transform all points to camera frame using numpy for speed
            arr = np.fromiter(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            if arr.size == 0:
                pts_cam = []
            else:
                xyz = np.vstack((arr['x'], arr['y'], arr['z'], np.ones(arr.size, dtype=np.float32)))  # 4xN
                cam = T.dot(xyz)  # 4xN
                pts_cam = list(zip(cam[0, :].tolist(), cam[1, :].tolist(), cam[2, :].tolist()))

        if not pts_cam:
            rospy.loginfo('paired_detect_and_depth: point cloud empty after transform')
            return

        # Compute distances for each detection by angular windowing in camera frame (equirect model)
        H, W = cv_img.shape[:2]
        win_yaw = math.radians(self.angle_window_deg)
        win_pitch = math.radians(max(0.5 * self.angle_window_deg, 1.0))  # a bit wider vertically

        results = []  # (label, u, v, dist_m)
        for label, u, v in dets:
            yaw, pitch = self._equirect_ray(u, v, W, H)
            dists = []
            for (x, y, z) in pts_cam:
                # Direction angles of point from camera
                pyaw = math.atan2(y, x)
                horiz = math.hypot(x, y)
                ppitch = math.atan2(z, max(horiz, 1e-6))
                if self._ang_diff(pyaw, yaw) <= win_yaw and abs(ppitch - pitch) <= win_pitch:
                    dists.append(math.sqrt(x * x + y * y + z * z))
            if dists:
                dist = float(np.median(dists))
            else:
                dist = float('nan')
            results.append((label, u, v, dist))

        # Append to object_list.txt
        try:
            with open(self.object_list_path, 'a') as f:
                for (label, u, v, dist) in results:
                    f.write(f"{stamp.to_sec():.3f}\t{label}\t{u:.1f}\t{v:.1f}\t{dist if not math.isnan(dist) else -1}\n")
            rospy.loginfo('paired_detect_and_depth: wrote %d detections to %s', len(results), self.object_list_path)
        except Exception as e:
            rospy.logwarn('Failed writing object list: %s', str(e))

    @staticmethod
    def _tf_to_matrix(trans, rot):
        tx, ty, tz = trans
        qx, qy, qz, qw = rot
        # Convert quaternion to rotation matrix
        x2 = qx + qx; y2 = qy + qy; z2 = qz + qz
        xx = qx * x2; yy = qy * y2; zz = qz * z2
        xy = qx * y2; xz = qx * z2; yz = qy * z2
        wx = qw * x2; wy = qw * y2; wz = qw * z2
        R = np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy, tx],
            [xy + wz, 1.0 - (xx + zz), yz - wx, ty],
            [xz - wy, yz + wx, 1.0 - (xx + yy), tz],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        return R

    @staticmethod
    def _equirect_ray(u, v, W, H):
        # Map pixel to yaw [-pi, pi] and pitch [-pi/2, pi/2]
        yaw = 2.0 * math.pi * (u / max(W, 1.0) - 0.5)
        pitch = math.pi * (0.5 - v / max(H, 1.0))
        return yaw, pitch

    @staticmethod
    def _ang_diff(a, b):
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return abs(d)


def main():
    rospy.init_node('paired_detect_and_depth', anonymous=False)
    _ = PairedDetectAndDepth()
    rospy.spin()


if __name__ == '__main__':
    main()

# ---------- Helpers (copied from camera_save_and_detect) ----------
def _bytes_to_array(msg, channels):
    width = msg.width
    height = msg.height
    step = msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if step == width * channels:
        return buf.reshape((height, width, channels))
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
        if enc.startswith('rgba'):
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    if enc in ('mono8', '8uc1'):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def rosimg_to_cv2_bgr(msg):
    enc = (msg.encoding or '').lower()
    if enc in ('mono8', '8uc1'):
        img = _mono_bytes_to_array(msg)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = _bytes_to_array(msg, 3)
    return _convert_encoding_to_bgr(img, enc)
