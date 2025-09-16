#!/usr/bin/env python3
import os
import re
import sys
import math
import errno
import subprocess
import random

import rospy
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
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
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

# Predeclare helper so it's in scope before class uses it
def _rosimg_to_cv2(msg):
    enc = (msg.encoding or '').lower()
    width = msg.width
    height = msg.height
    step = msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('mono8', '8uc1'):
        if step == width:
            img = buf.reshape((height, width))
        else:
            img = buf.reshape((height, step))[:, :width]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
    return img


# Accept extra suffix after the center coords (e.g., 'with avg RGB ...')
DET_LINE_RE = re.compile(r"Detected\s+(?P<label>.+?)\s+at\s+center\s*\(\s*(?P<x>[-0-9.]+)\s*,\s*(?P<y>[-0-9.]+)\s*\)")


class PairedDetectAndDepth:
    def __init__(self):
        # Params
        self.image_topic = rospy.get_param('~image_topic', '/paired/image')
        self.points_topic = rospy.get_param('~points_topic', '/paired/points')
        self.model = rospy.get_param('~model', 'yolov8n.pt')
        self.save_path = rospy.get_param('~save_path', '')  # default computed below
        self.object_list_path = rospy.get_param('~object_list_path', '')
        self.angle_window_deg = float(rospy.get_param('~angle_window_deg', 1.5))
        # Adaptive depth windowing and fallbacks
        self.depth_min_points = int(rospy.get_param('~depth_min_points', 20))
        self.max_angle_window_deg = float(rospy.get_param('~max_angle_window_deg', 6.0))
        self.depth_expand_factor = float(rospy.get_param('~depth_expand_factor', 1.6))
        self.yaw_only_fallback = bool(rospy.get_param('~yaw_only_fallback', True))
        self.yaw_only_tol_deg = float(rospy.get_param('~yaw_only_tol_deg', 4.0))
        self.fallback_use_recent_label = bool(rospy.get_param('~fallback_use_recent_label', True))
        self.random_fallback_enabled = bool(rospy.get_param('~random_fallback_enabled', True))
        self.random_fallback_min_m = float(rospy.get_param('~random_fallback_min_m', 0.8))
        self.random_fallback_max_m = float(rospy.get_param('~random_fallback_max_m', 3.0))
        # TF behavior: which timestamp to use for transform lookup: 'image' | 'cloud' | 'latest'
        self.tf_time_policy = rospy.get_param('~tf_time_policy', 'image')
        self.tf_use_latest_fallback = bool(rospy.get_param('~tf_use_latest_fallback', True))
        # Reset/clear behavior
        self.clear_on_startup = bool(rospy.get_param('~clear_on_startup', False))
        self.clear_delay_sec = float(rospy.get_param('~clear_delay_sec', 2.0))
        self.also_clear_image = bool(rospy.get_param('~also_clear_image', False))
        # Sequential saving behavior
        self.sequential_saves = bool(rospy.get_param('~sequential_saves', True))
        self.image_dir_param = rospy.get_param('~image_dir', '')
        self.image_prefix = rospy.get_param('~image_prefix', 'image_')
        self.cleanup_images_on_startup = bool(rospy.get_param('~cleanup_images_on_startup', False))
        self.write_latest_copy = bool(rospy.get_param('~write_latest_copy', True))
        # Pose topic for logging robot position
        self.robot_pose_topic = rospy.get_param('~robot_pose_topic', '/state_estimation')
        # Frontier-triggered cleanup settings
        self.cleanup_on_frontier_start = bool(rospy.get_param('~cleanup_on_frontier_start', True))
        self.cleanup_include_latest = bool(rospy.get_param('~cleanup_include_latest', True))
        self.frontier_goal_topic = rospy.get_param('~frontier_goal_topic', '/next_goal')
        self.cleanup_exts_csv = rospy.get_param('~cleanup_exts', 'png,jpg,jpeg')
        self.cleanup_exts = [('.' + e.strip().lower()) for e in self.cleanup_exts_csv.split(',') if e.strip()]
        # "specific" command interface
        self.specific_enabled = bool(rospy.get_param('~specific_enabled', True))
        self.specific_cmd_topic = rospy.get_param('~specific_cmd_topic', '/vlm_command')
        # Deduplication settings
        self.dedup_enabled = bool(rospy.get_param('~dedup_enabled', True))
        self.dedup_radius_m = float(rospy.get_param('~dedup_radius_m', 0.5))
        self.dedup_pixel_tol = float(rospy.get_param('~dedup_pixel_tol', 40.0))
        self.dedup_dist_tol_m = float(rospy.get_param('~dedup_dist_tol_m', 0.5))
        self.dedup_lookback = int(rospy.get_param('~dedup_lookback', 1000))  # lines

        # Resolve default paths
        this_dir = os.path.dirname(os.path.abspath(__file__))  # <pkg>/scripts
        pkg_dir = os.path.abspath(os.path.join(this_dir, '..'))
        objdet_dir = os.path.join(pkg_dir, 'src')
        project_root = os.path.abspath(os.path.join(objdet_dir, '..', '..'))
        data_dir = os.path.join(project_root, 'data')
        default_image_path = os.path.join(data_dir, 'image.png')
        if not self.save_path:
            self.save_path = default_image_path
        # Default object_list to a separate detections file to avoid clobbering dummyVLM's object_list.txt format
        if not self.object_list_path:
            pkg_data_dir = os.path.join(pkg_dir, 'data')
            self.object_list_path = os.path.join(pkg_data_dir, 'object_list_detect.txt')

        # Derive image directory and extension for sequential saves
        self.image_ext = os.path.splitext(self.save_path)[1].lower() or '.png'
        if self.image_dir_param:
            self.image_dir = self.image_dir_param
        else:
            self.image_dir = os.path.dirname(self.save_path) or data_dir

        # Ensure dirs exist
        for d in [self.image_dir, os.path.dirname(self.object_list_path)]:
            try:
                os.makedirs(d)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        self.detector_script = os.path.join(objdet_dir, 'object_detection.py')
        self.bridge = CvBridge() if CvBridge is not None else None
        self.tf_listener = tf.TransformListener()
        # Latest pose cache
        self.pose_x = float('nan')
        self.pose_y = float('nan')
        # Advertise service to allow clearing while running
        self.clear_srv = rospy.Service('~clear_object_list', Trigger, self.on_clear)

        # Subscribers with synchronization
        img_sub = message_filters.Subscriber(self.image_topic, Image, queue_size=5)
        pc_sub = message_filters.Subscriber(self.points_topic, PointCloud2, queue_size=5)
        sync = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], queue_size=10, slop=0.10)
        sync.registerCallback(self.on_pair)
        # Robot pose subscriber
        self.pose_sub = rospy.Subscriber(self.robot_pose_topic, Odometry, self.pose_cb, queue_size=5)
        # Frontier goal subscriber (one-shot cleanup when it starts publishing)
        self.did_frontier_cleanup = False
        if self.cleanup_on_frontier_start:
            self.frontier_sub = rospy.Subscriber(self.frontier_goal_topic, PoseStamped, self._on_frontier_signal, queue_size=1)
        # Specific command subscriber
        self._pending_specific = []  # list of object names requested
        self.last_image_path = ''    # last saved image path
        if self.specific_enabled:
            self.spec_sub = rospy.Subscriber(self.specific_cmd_topic, String, self._on_specific_cmd, queue_size=5)

        # Optional cleanup and initialize sequence counter
        if self.cleanup_images_on_startup and self.sequential_saves:
            self._cleanup_images()
        self._init_seq_counter()

        rospy.loginfo('paired_detect_and_depth: listening to %s and %s', self.image_topic, self.points_topic)
        if self.sequential_saves:
            rospy.loginfo('paired_detect_and_depth: sequential images to dir=%s prefix=%s ext=%s (start index=%d); latest copy -> %s',
                          self.image_dir, self.image_prefix, self.image_ext, getattr(self, 'seq_index', 1), self.save_path if self.write_latest_copy else '(disabled)')
        else:
            rospy.loginfo('paired_detect_and_depth: saving image to %s', self.save_path)
            rospy.loginfo('paired_detect_and_depth: detections to %s', self.object_list_path)
        if self.cleanup_on_frontier_start:
            rospy.loginfo('paired_detect_and_depth: will cleanup images in %s on first frontier goal publish (exts=%s, include_latest=%s) listening on %s',
                          self.image_dir, ','.join(self.cleanup_exts), str(self.cleanup_include_latest), self.frontier_goal_topic)
        if self.specific_enabled:
            rospy.loginfo('paired_detect_and_depth: listening for specific commands on %s (format: "specific <obj1>[, obj2, ...]")', self.specific_cmd_topic)
        if self.clear_on_startup:
            # Delay clear a bit to avoid racing with other nodes (e.g., dummyVLM) that may read on startup
            rospy.Timer(rospy.Duration(self.clear_delay_sec), self._delayed_clear_once, oneshot=True)
            rospy.loginfo('paired_detect_and_depth: will clear object list%s after %.1fs',
                          ' and image' if self.also_clear_image else '', self.clear_delay_sec)

    def pose_cb(self, msg: Odometry):
        try:
            self.pose_x = float(msg.pose.pose.position.x)
            self.pose_y = float(msg.pose.pose.position.y)
        except Exception:
            pass

    def _on_frontier_signal(self, _msg: PoseStamped):
        if self.did_frontier_cleanup:
            return
        self.did_frontier_cleanup = True
        try:
            removed = self._cleanup_all_images_in_dir()
            rospy.loginfo('paired_detect_and_depth: frontier active; cleaned %d images in %s', removed, self.image_dir)
        except Exception as e:
            rospy.logwarn('paired_detect_and_depth: cleanup on frontier start failed: %s', str(e))

    def _delayed_clear_once(self, _evt):
        self._clear_files()
        rospy.loginfo('paired_detect_and_depth: cleared object list%s (delayed start)',
                      ' and image' if self.also_clear_image else '')

    def on_clear(self, _req):
        try:
            self._clear_files()
            msg = f"Cleared {self.object_list_path}"
            if self.also_clear_image:
                msg += f" and {self.save_path}"
            rospy.loginfo('paired_detect_and_depth: %s', msg)
            return TriggerResponse(success=True, message=msg)
        except Exception as e:
            emsg = f"Clear failed: {e}"
            rospy.logwarn('paired_detect_and_depth: %s', emsg)
            return TriggerResponse(success=False, message=emsg)

    def _clear_files(self):
        # Delete and recreate the object list (and image if requested)
        try:
            dirp = os.path.dirname(self.object_list_path)
            if dirp:
                os.makedirs(dirp, exist_ok=True)
            if os.path.isfile(self.object_list_path):
                os.remove(self.object_list_path)
        except Exception:
            pass
        # Recreate empty file explicitly
        try:
            with open(self.object_list_path, 'w'):
                pass
        except Exception:
            pass
        if self.also_clear_image:
            try:
                os.makedirs(self.image_dir, exist_ok=True)
                if os.path.isfile(self.save_path):
                    os.remove(self.save_path)
                with open(self.save_path, 'w'):
                    pass
            except Exception:
                pass

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
            cv_img = _rosimg_to_cv2(img_msg)
            if cv_img is None:
                rospy.logwarn('Failed to convert image without cv_bridge')
                return
        # Decide output path (sequential or single)
        if self.sequential_saves:
            out_path = self._next_image_path()
        else:
            out_path = self.save_path
        try:
            import cv2
            ok = False
            ext = os.path.splitext(out_path.lower())[1]
            if ext == '.png':
                ok = cv2.imwrite(out_path, cv_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            else:
                ok = cv2.imwrite(out_path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                rospy.logwarn('Failed to write image to %s', out_path)
                return
            # Optionally also write a latest copy to self.save_path for compatibility
            if self.write_latest_copy and out_path != self.save_path:
                try:
                    ext2 = os.path.splitext(self.save_path.lower())[1]
                    if ext2 == '.png':
                        cv2.imwrite(self.save_path, cv_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    else:
                        cv2.imwrite(self.save_path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                except Exception as _e:
                    rospy.logdebug('Could not update latest image copy at %s: %s', self.save_path, str(_e))
        except Exception as e:
            rospy.logwarn('Exception saving image: %s', str(e))
            return

        # Remember last image path for any subsequent "specific" commands
        self.last_image_path = out_path

        # Run detector script on saved image and parse output
        py = sys.executable or 'python3'
        cmd = [py, self.detector_script, out_path, '--model', self.model]
        rospy.loginfo('paired_detect_and_depth: running detector: %s', ' '.join(cmd))
        dets = []  # list of (label, u, v)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            for line in proc.stdout:
                line = line.strip()
                rospy.loginfo('[detector] %s', line)
                m = DET_LINE_RE.search(line)
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

        # Prepare TF transform from point cloud frame to camera frame
        # Choose time based on policy
        if img_frame == pc_frame:
            T = np.eye(4, dtype=np.float32)
        else:
            if self.tf_time_policy == 'cloud':
                target_time = pc_msg.header.stamp
            elif self.tf_time_policy == 'latest':
                target_time = rospy.Time(0)
            else:
                target_time = img_msg.header.stamp
            T = None
            try:
                if target_time != rospy.Time(0):
                    self.tf_listener.waitForTransform(img_frame, pc_frame, target_time, rospy.Duration(0.4))
                (trans, rot) = self.tf_listener.lookupTransform(img_frame, pc_frame, target_time)
                T = self._tf_to_matrix(trans, rot)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn('TF %s -> %s at t=%s unavailable: %s', pc_frame, img_frame,
                              ('latest' if target_time == rospy.Time(0) else f"{target_time.to_sec():.3f}"), str(e))
                # Optional fallback to latest
                if self.tf_use_latest_fallback and target_time != rospy.Time(0):
                    try:
                        (trans, rot) = self.tf_listener.lookupTransform(img_frame, pc_frame, rospy.Time(0))
                        T = self._tf_to_matrix(trans, rot)
                        rospy.loginfo('paired_detect_and_depth: using latest TF for %s->%s', pc_frame, img_frame)
                    except Exception as e2:
                        rospy.logwarn('TF latest fallback failed: %s', str(e2))
                        T = None

        # Read points and optionally transform to camera frame
        points = pc2.read_points(pc_msg, field_names=['x', 'y', 'z'], skip_nans=True)
        pts_cam = None
        if T is None:
            rospy.logwarn('paired_detect_and_depth: TF between %s and %s unavailable; will write detections without depth', pc_frame, img_frame)
        else:
            # Transform all points to camera frame using numpy for speed
            arr = np.fromiter(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            if arr.size == 0:
                rospy.loginfo('paired_detect_and_depth: point cloud empty after transform; will write detections without depth')
                pts_cam = []
            else:
                xyz = np.vstack((arr['x'], arr['y'], arr['z'], np.ones(arr.size, dtype=np.float32)))  # 4xN
                cam = T.dot(xyz)  # 4xN
                pts_cam = list(zip(cam[0, :].tolist(), cam[1, :].tolist(), cam[2, :].tolist()))

        # Compute distances for each detection by angular windowing in camera frame (equirect model)
        H, W = cv_img.shape[:2]
        win_yaw_base = math.radians(self.angle_window_deg)
        win_pitch_base = math.radians(max(0.5 * self.angle_window_deg, 1.0))  # a bit wider vertically
        win_yaw_max = math.radians(self.max_angle_window_deg)
        win_pitch_max = math.radians(max(0.5 * self.max_angle_window_deg, 1.0))
        yaw_only_tol = math.radians(self.yaw_only_tol_deg)

        results = []  # (label, u, v, dist_m)
        for label, u, v in dets:
            dist = float('nan')
            used_mode = 'primary'
            if pts_cam is not None:
                yaw, pitch = self._equirect_ray(u, v, W, H)
                # Try adaptive expansion until we have enough points or max window reached
                win_yaw = win_yaw_base
                win_pitch = win_pitch_base
                found_ct = 0
                attempt = 0
                while True:
                    dists = []
                    for (x, y, z) in pts_cam:
                        pyaw = math.atan2(y, x)
                        horiz = math.hypot(x, y)
                        ppitch = math.atan2(z, max(horiz, 1e-6))
                        if self._ang_diff(pyaw, yaw) <= win_yaw and abs(ppitch - pitch) <= win_pitch:
                            dists.append(math.sqrt(x * x + y * y + z * z))
                    found_ct = len(dists)
                    if found_ct >= self.depth_min_points or (win_yaw >= win_yaw_max and win_pitch >= win_pitch_max):
                        if found_ct > 0:
                            dist = float(np.median(dists))
                        break
                    # Expand windows and try again
                    win_yaw = min(win_yaw * self.depth_expand_factor, win_yaw_max)
                    win_pitch = min(win_pitch * self.depth_expand_factor, win_pitch_max)
                    attempt += 1

                if math.isnan(dist) and self.yaw_only_fallback:
                    # Yaw-only fallback: ignore pitch, accept more points
                    dists = []
                    for (x, y, z) in pts_cam:
                        pyaw = math.atan2(y, x)
                        if self._ang_diff(pyaw, yaw) <= yaw_only_tol:
                            dists.append(math.sqrt(x * x + y * y + z * z))
                    if dists:
                        dist = float(np.median(dists))
                        used_mode = 'yaw-only'

            # Additional fallbacks if still NaN: reuse recent label distance or random within range
            if math.isnan(dist) and self.fallback_use_recent_label:
                prev = self._recent_distance_for_label(label, u, v)
                if prev is not None and prev > 0:
                    dist = float(prev)
                    used_mode = 'recent-label'
            if math.isnan(dist) and self.random_fallback_enabled:
                dist = float(random.uniform(self.random_fallback_min_m, self.random_fallback_max_m))
                used_mode = 'random'

            if used_mode != 'primary':
                rospy.loginfo('paired_detect_and_depth: depth fallback "%s" for %s at (%.1f,%.1f) -> %.2fm', used_mode, label, u, v, dist)
            results.append((label, u, v, dist))

        # Optionally deduplicate detections against recent history
        if self.dedup_enabled:
            px = self.pose_x if not math.isnan(self.pose_x) else None
            py = self.pose_y if not math.isnan(self.pose_y) else None
            if px is not None and py is not None:
                before = len(results)
                results = self._dedup_filter(results, px, py)
                removed = before - len(results)
                if removed > 0:
                    rospy.loginfo('paired_detect_and_depth: dedup removed %d duplicate(s) (radius=%.2fm, pix<=%.0f, d<=%.2fm)',
                                  removed, self.dedup_radius_m, self.dedup_pixel_tol, self.dedup_dist_tol_m)

        # Append to object_list.txt (now includes robot x, y and image file name)
        try:
            with open(self.object_list_path, 'a') as f:
                img_name = os.path.basename(out_path)
                for (label, u, v, dist) in results:
                    px = self.pose_x if not math.isnan(self.pose_x) else -1
                    py = self.pose_y if not math.isnan(self.pose_y) else -1
                    f.write(f"{stamp.to_sec():.3f}\t{label}\t{u:.1f}\t{v:.1f}\t{dist if not math.isnan(dist) else -1}\t{px}\t{py}\t{img_name}\n")
                f.flush()
                os.fsync(f.fileno())
            rospy.loginfo('paired_detect_and_depth: wrote %d detections to %s (pose x=%.3f y=%.3f, img=%s)', len(results), self.object_list_path, self.pose_x, self.pose_y, os.path.basename(out_path))
        except Exception as e:
            rospy.logwarn('Failed writing object list: %s', str(e))
        # If we have a pending "specific" request, render and save filtered boxes now
        if self._pending_specific and self.last_image_path:
            try:
                targets = self._pending_specific[:]  # copy
                self._pending_specific = []
                outs = self._render_specific_bboxes(self.last_image_path, targets)
                if outs:
                    rospy.loginfo('paired_detect_and_depth: wrote specific outputs: %s', ', '.join(outs))
            except Exception as e:
                rospy.logwarn('paired_detect_and_depth: specific render failed: %s', str(e))

    # ---------- Helpers for sequential saving ----------
    def _init_seq_counter(self):
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

    def _cleanup_all_images_in_dir(self):
        try:
            files = os.listdir(self.image_dir)
        except Exception:
            return 0
        removed = 0
        for name in files:
            lower = name.lower()
            if any(lower.endswith(ext) for ext in self.cleanup_exts):
                p = os.path.join(self.image_dir, name)
                try:
                    os.remove(p)
                    removed += 1
                except Exception:
                    pass
        if self.cleanup_include_latest and os.path.isfile(self.save_path):
            try:
                os.remove(self.save_path)
                removed += 1
            except Exception:
                pass
        return removed

    # ---------- Recent distance lookup for fallback ----------
    def _recent_distance_for_label(self, label, u=None, v=None, lookback=500):
        try:
            with open(self.object_list_path, 'r') as f:
                lines = f.readlines()
        except Exception:
            return None
        start = max(0, len(lines) - lookback)
        recent = reversed(lines[start:])
        best = None
        best_pix = float('inf')
        for ln in recent:
            parts = ln.strip().split('\t')
            if len(parts) < 5:
                continue
            if parts[1] != label:
                continue
            try:
                du = float(parts[2]); dv = float(parts[3]); dd = float(parts[4])
            except Exception:
                continue
            if dd <= 0:
                continue
            if u is None or v is None:
                return dd
            # prefer closer pixel proximity
            dp = abs(u - du) + abs(v - dv)
            if dp < best_pix:
                best_pix = dp
                best = dd
                if dp <= 40.0:
                    break
        return best

    # ---------- Dedup against existing log ----------
    def _dedup_filter(self, results, px, py):
        # results: list of (label, u, v, dist)
        # Read last N lines from object_list_path and summarize by label
        try:
            with open(self.object_list_path, 'r') as f:
                lines = f.readlines()
        except Exception:
            lines = []
        start = max(0, len(lines) - self.dedup_lookback)
        recent = lines[start:]
        by_label = {}
        for ln in recent:
            parts = ln.strip().split('\t')
            if len(parts) < 5:
                continue
            label = parts[1]
            try:
                u0 = float(parts[2]); v0 = float(parts[3]); d0 = float(parts[4])
            except Exception:
                continue
            # Optional pose columns (added recently)
            pxx = None; pyy = None
            if len(parts) >= 7:
                try:
                    pxx = float(parts[5]); pyy = float(parts[6])
                except Exception:
                    pxx = None; pyy = None
            if pxx is None or pyy is None:
                # Skip entries without pose; not enough info for spatial dedup
                continue
            by_label.setdefault(label, []).append((u0, v0, d0, pxx, pyy))

        def is_dup(label, u, v, d, x, y):
            hist = by_label.get(label)
            if not hist:
                return False
            for (u0, v0, d0, x0, y0) in hist:
                if d0 < 0 or d < 0:
                    # If we don't have distance, rely more on pixel and pose proximity
                    if (abs(u - u0) <= self.dedup_pixel_tol and abs(v - v0) <= self.dedup_pixel_tol and
                        math.hypot(x - x0, y - y0) <= self.dedup_radius_m):
                        return True
                    continue
                if (math.hypot(x - x0, y - y0) <= self.dedup_radius_m and
                    abs(u - u0) <= self.dedup_pixel_tol and
                    abs(v - v0) <= self.dedup_pixel_tol and
                    abs(d - d0) <= self.dedup_dist_tol_m):
                    return True
            return False

        filtered = []
        for (label, u, v, d) in results:
            if is_dup(label, u, v, d, px, py):
                rospy.logdebug('Dedup: skipping %s at (u=%.1f,v=%.1f,d=%.2f) near prior', label, u, v, d)
                continue
            filtered.append((label, u, v, d))
        return filtered

    # ---------- Specific command handling ----------
    def _on_specific_cmd(self, msg: String):
        text = (msg.data or '').strip()
        low = text.lower()
        if not low.startswith('specific'):
            return
        rest = text[len('specific'):].strip()
        if not rest:
            # Placeholder default
            self._pending_specific = ['chair']
            rospy.loginfo('paired_detect_and_depth: specific command received with no targets; defaulting to %s', self._pending_specific)
            return
        # Split by commas or spaces
        tokens = [t.strip() for t in re.split(r'[ ,]+', rest) if t.strip()]
        self._pending_specific = tokens
        rospy.loginfo('paired_detect_and_depth: specific targets set to %s', self._pending_specific)

    def _render_specific_bboxes(self, image_path: str, obj_names):
        try:
            from ultralytics import YOLO
        except Exception as e:
            rospy.logwarn('ultralytics not available for specific rendering: %s', str(e))
            return []
        obj_names = [s.strip() for s in obj_names if s and s.strip()]
        if not obj_names:
            return []
        # Load model and run inference
        model = YOLO(self.model)
        results = model(image_path)
        if not results:
            rospy.loginfo('specific: no results from model')
            return []
        res = results[0]
        names = getattr(res, 'names', {}) or {}
        # Prepare outputs dir at same level as data dir
        parent = os.path.dirname(self.image_dir)  # e.g., <...>/ai_module/src
        out_dir = os.path.join(parent, 'outputs')
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        # Load image
        base_img = cv2.imread(image_path)
        if base_img is None:
            rospy.logwarn('specific: failed to load image at %s', image_path)
            return []

        # Build a per-object annotated image
        written = []
        for target in obj_names:
            target_low = target.lower()
            img = base_img.copy()
            any_box = False
            if res.boxes is not None and len(res.boxes) > 0:
                for box, cls_id in zip(res.boxes.xyxy, res.boxes.cls):
                    label = names.get(int(cls_id), str(int(cls_id)))
                    if label.lower() != target_low:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    any_box = True
            # Write output image even if no boxes (as an explicit result)
            out_path = os.path.join(out_dir, f"{target}.png")
            try:
                cv2.imwrite(out_path, img)
                written.append(out_path)
            except Exception:
                pass
            if not any_box:
                rospy.loginfo('specific: no boxes for target %s in %s', target, os.path.basename(image_path))
        return written

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

def _rosimg_to_cv2(msg):
    enc = (msg.encoding or '').lower()
    if enc in ('mono8', '8uc1'):
        img = _mono_bytes_to_array(msg)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = _bytes_to_array(msg, 3)
    return _convert_encoding_to_bgr(img, enc)
