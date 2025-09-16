#!/usr/bin/env python3
import os
import sys
import subprocess
import errno

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class CameraSaveAndDetect:
    def __init__(self):
        self.image_topic = rospy.get_param('~image_topic', '/camera/image')
        self.interval_sec = float(rospy.get_param('~interval_sec', 10.0))
        self.model = rospy.get_param('~model', 'yolov8n.pt')
        self.run_if_no_subscribers = bool(rospy.get_param('~run_if_no_subscribers', True))

        # Compute save path to match object_detection.py's default
        # object_detection.py lives in <pkg_dir>/src/object_detection.py
        this_dir = os.path.dirname(os.path.abspath(__file__))  # <pkg_dir>/scripts
        pkg_dir = os.path.abspath(os.path.join(this_dir, '..'))
        objdet_dir = os.path.join(pkg_dir, 'src')
        project_root = os.path.abspath(os.path.join(objdet_dir, '..', '..'))  # mimics object_detection.py logic
        data_dir = os.path.join(project_root, 'data')
        self.save_path = os.path.join(data_dir, 'image.jpg')
        self.detector_script = os.path.join(objdet_dir, 'object_detection.py')

        # Ensure data dir exists
        try:
            os.makedirs(data_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.bridge = CvBridge()
        self.last_save_time = rospy.Time(0)

        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        rospy.loginfo('camera_save_and_detect: listening to %s, interval %.2fs, saving to %s',
                      self.image_topic, self.interval_sec, self.save_path)

    def image_cb(self, msg: Image):
        now = msg.header.stamp if msg.header.stamp and msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        if self.last_save_time.to_sec() > 0 and (now - self.last_save_time).to_sec() < self.interval_sec:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logwarn('cv_bridge bgr8 conversion failed (%s); trying passthrough', str(e))
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Convert single-channel to BGR for JPEG write consistency
                if len(cv_img.shape) == 2:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            except CvBridgeError as e2:
                rospy.logwarn('cv_bridge passthrough failed: %s', str(e2))
                return

        # Save as JPEG
        try:
            ok = cv2.imwrite(self.save_path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                rospy.logwarn('Failed to write image to %s', self.save_path)
                return
        except Exception as e:
            rospy.logwarn('Exception writing image: %s', str(e))
            return

        self.last_save_time = now
        rospy.loginfo('Saved image to %s; running detector...', self.save_path)

        # Run detector as a subprocess
        py = sys.executable or 'python3'
        cmd = [py, self.detector_script, self.save_path, '--model', self.model]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            # Stream output to ROS log
            for line in proc.stdout:
                rospy.loginfo('[detector] %s', line.strip())
            rc = proc.wait()
            rospy.loginfo('Detector exited with code %d', rc)
        except Exception as e:
            rospy.logwarn('Failed to run detector: %s', str(e))


def main():
    rospy.init_node('camera_save_and_detect', anonymous=False)
    node = CameraSaveAndDetect()
    rospy.spin()


if __name__ == '__main__':
    main()
