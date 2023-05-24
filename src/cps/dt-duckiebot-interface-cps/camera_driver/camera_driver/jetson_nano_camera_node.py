#!/usr/bin/env python3

import imp
import cv2
import time
import rclpy
import psutil
import numpy as np

from threading import Thread
from typing import Tuple, cast
from collections import namedtuple
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage

from camera_driver.AbsCameraNode import (
    AbsCameraNode,
    DeclareParams,
)


CameraMode = namedtuple('CameraMode', 'id width height fps fov')


class JetsonNanoCameraNode(AbsCameraNode):
    """Handles the imagery on a Jetson Nano."""

    # each mode defines [width, height, fps]
    CAMERA_MODES = [
        CameraMode(0, 3264, 2464, 21, 'full'),
        CameraMode(1, 3264, 1848, 28, 'partial'),
        CameraMode(2, 1920, 1080, 30, 'partial'),
        CameraMode(3, 1640, 1232, 30, 'full'),
        CameraMode(4, 1280, 720, 60, 'partial'),
        CameraMode(5, 1280, 720, 120, 'partial'),
    ]
    # exposure time range (ns)
    EXPOSURE_TIMERANGES = {
        "sports": [100000, 80000000],
        "night":  [100000, 1000000000]
    }
    DEFAULT_EXPOSURE_MODE = "sports"

    ADDITIONAL_PARAMS = [
        DeclareParams("allow_partial_fov", Parameter.Type.BOOL, False),
        DeclareParams("use_hw_acceleration", Parameter.Type.BOOL, False)
    ]

    def __init__(self, node_name: str):
        super(JetsonNanoCameraNode, self).__init__(node_name,
            JetsonNanoCameraNode.ADDITIONAL_PARAMS)

        # prepare gstreamer pipeline
        self._device = None
        # prepare data flow monitor
        self._flow_monitor = Thread(target=self._flow_monitor_fcn)
        self._flow_monitor.setDaemon(True)
        self._flow_monitor.start()

        self.get_logger().info("Node initialized.")

    def _flow_monitor_fcn(self):
        i = 0
        failure_timeout = 20
        while not self.is_shutdown:
            # do nothing for the first `sleep_until` seconds,
            # then check every 5 seconds
            if (self._has_published or i > failure_timeout) and i % 5 == 0:
                elapsed_since_last = time.time() - self._last_image_published_time
                # reset nvargus if no images were received within the last 10 secs
                if elapsed_since_last >= 10:
                    self.get_logger().info(
                        f"[data-flow-monitor]: Detected a period of"\
                        f" {int(elapsed_since_last)} seconds during which no"\
                        f" images were produced, restarting camera process.")
                    # find PID of the nvargus process
                    killed = False
                    for proc in psutil.process_iter():
                        try:
                            # get process name and pid from process object
                            process_cmdline = proc.cmdline()
                            if len(process_cmdline) <= 0:
                                continue
                            process_bin = process_cmdline[0]
                            # if it is not the one we are looking for, skip
                            if not process_bin.startswith("/usr/sbin/nvargus-daemon"):
                                continue
                            # this is the one we are looking for
                            self.get_logger().info(f"[data-flow-monitor]: Process 'nvargus-daemon' found "
                                         f"with PID #{proc.pid}")
                            # - kill nvargus
                            self.get_logger().info(f"[data-flow-monitor]: Killing nvargus.")
                            proc.kill()
                            time.sleep(1)
                            # - stop camera node, then wait 10 seconds
                            self.get_logger().info(f"[data-flow-monitor]: Clearing camera...")
                            self.stop(force=True)
                            self.get_logger().info(f"[data-flow-monitor]: Camera cleared. Rebooting.")
                            # - exit camera node, roslaunch will respawn in 10 seconds
                            # rospy.signal_shutdown("Data flow monitor has closed the node")
                            self.get_logger().info("Data flow monitor has closed the node")
                            time.sleep(1)
                            exit(1)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            pass
                    if not killed:
                        self.loginfo("[data-flow-monitor]: Process 'nvargus-daemon' not found.")
            # ---
            i += 1
            time.sleep(1)

    def run(self):
        """ Image capture procedure.

            Captures a frame from the /dev/video2 image sink and publishes it.
        """
        if self._device is None or not self._device.isOpened():
            self.logerr('Device was found closed')
            return
        # get first frame
        retval, image = self._device.read() if self._device else (False, None)
        # keep reading
        while (not self.is_stopped) and (not self.is_shutdown) and retval:
            if image is not None:
                if self._use_hw_acceleration:
                    # with HW acceleration, the NVJPG Engine module is used to encode RGB -> JPEG
                    image = cast(np.ndarray, image)
                    image_msg = CompressedImage()
                    image_msg.header.stamp = self.get_clock().now().to_msg()
                    image_msg.format = "jpeg"
                    image_msg.data = image.tobytes()
                else:
                    # without HW acceleration, the image is returned as RGB, encode on CPU
                    if image is not None:
                        image = np.uint8(image)
                    image_msg = self._bridge.cv2_to_compressed_imgmsg(image, dst_format='jpeg')
                # publish the compressed image
                self.publish(image_msg)
            # grab next frame
            retval, image = self._device.read() if self._device else (False, None)
        self.get_logger().info('Camera worker stopped.')

    def setup(self):
        if self._device is None:
            self._device = cv2.VideoCapture()
        # check if the device is opened
        if self._device is None:
            msg = "OpenCV cannot open gstreamer resource"
            self.get_logger().error(msg)
            raise RuntimeError(msg)
        # open the device
        if not self._device.isOpened():
            try:
                self._device.open(self.gst_pipeline_string(), cv2.CAP_GSTREAMER)
                # make sure the device is open
                if not self._device.isOpened():
                    msg = "OpenCV cannot open gstreamer resource"
                    self.get_logger().error(msg)
                    raise RuntimeError(msg)
                # try getting a sample image
                retval, _ = self._device.read()
                if not retval:
                    msg = "Could not read image from camera"
                    self.get_logger().error(msg)
                    raise RuntimeError(msg)
            except (Exception, RuntimeError):
                self.stop()
                msg = "Could not start camera"
                self.get_logger().error(msg)
                raise RuntimeError(msg)

    def release(self, force: bool = False):
        if self._device is not None:
            if force:
                self.get_logger().info('Forcing release of the GST pipeline...')
            else:
                self.get_logger().info('Releasing GST pipeline...')
                try:
                    self._device.release()
                except Exception:
                    pass
        self.get_logger().info('GST pipeline released.')
        self._device = None

    def gst_pipeline_string(self):
        res_w, res_h, fps = self._res_w, self._res_h, self._framerate
        fov = ('full', 'partial') if self._allow_partial_fov else ('full',)
        # find best mode
        camera_mode = self.get_mode(res_w, res_h, fps, fov)
        # cap frequency
        if fps > camera_mode.fps:
            self.get_logger().warn("Camera framerate({}fps) too high for the camera mode (max: {}fps), "
                         "capping at {}fps.".format(fps, camera_mode.fps, camera_mode.fps))
            fps = camera_mode.fps
        # get exposure time
        exposure_time = self.EXPOSURE_TIMERANGES.get(
            self._exposure_mode,
            self.EXPOSURE_TIMERANGES[self.DEFAULT_EXPOSURE_MODE]
        )
        # compile gst pipeline
        if self._use_hw_acceleration:
            gst_pipeline = """ \
                nvarguscamerasrc \
                sensor-mode={} exposuretimerange="{} {}" ! \
                video/x-raw(memory:NVMM), width={}, height={}, format=NV12, framerate={}/1 ! \
                nvjpegenc ! \
                appsink \
            """.format(
                camera_mode.id,
                *exposure_time,
                self._res_w,
                self._res_h,
                fps
            )
        else:
            gst_pipeline = """ \
                nvarguscamerasrc \
                sensor-mode={} exposuretimerange="{} {}" ! \
                video/x-raw(memory:NVMM), width={}, height={}, format=NV12, framerate={}/1 ! \
                nvvidconv ! 
                video/x-raw, format=BGRx ! 
                videoconvert ! \
                appsink \
            """.format(
                camera_mode.id,
                *exposure_time,
                self._res_w,
                self._res_h,
                fps
            )
        # ---
        self.get_logger().debug("Using GST pipeline: `{}`".format(gst_pipeline))
        return gst_pipeline

    def get_mode(self, width: int, height: int, fps: int, fov: Tuple[str]) -> CameraMode:
        candidates = {
            m for m in self.CAMERA_MODES
            if m.width >= width and m.height >= height and m.fps >= fps and m.fov in fov
        }.union({self.CAMERA_MODES[0]})
        return sorted(candidates, key=lambda m: m.id)[-1]


def main(args=None):
    rclpy.init(args=args)
    node = JetsonNanoCameraNode(node_name="jetson_camera_node")
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
