#!/usr/bin/env python3


import os
import cv2
import time
import copy
import numpy as np

from threading import Thread
from dataclasses import dataclass
from typing import Any, List, Optional
from abc import ABC, abstractmethod

from cv_bridge import CvBridge

from rclpy.node import Node
from rclpy.parameter import Parameter
# from sensor_msgs.srv import SetCameraInfo
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage, CameraInfo

from dt_calibration_utils import calibration_file


class CalibrationFileNotFoundException(Exception):
    pass


@dataclass
class DeclareParams:
    name: str
    type: Parameter.Type
    default_value: Any = None


class AbsCameraNode(ABC, Node):

    CALIBRATION_RELATIVE_FOLDER = 'camera_intrinsic'

    def __init__(self, node_name: str,
        params:  Optional[List[DeclareParams]]=None
    ):
        super().__init__(node_name)

        self._get_parameters(params)
        self._frame_id = self._veh + '/camera_optical_frame'
        self.add_on_set_parameters_callback(self._params_callback)

        # instrinsic calibration
        self._load_calibration_file()
        self.original_camera_info.header.frame_id = self._frame_id
        self.current_camera_info = copy.deepcopy(self.original_camera_info)

        # create cv bridge
        self._bridge = CvBridge()

        # Setup publishers
        self._has_published = False
        self._is_stopped = False
        self.is_shutdown = False
        self._worker = None
        self.pub_img = self.create_publisher(CompressedImage,
            "~/image/compressed", 1)
        self.pub_camera_info = self.create_publisher(CameraInfo,
            "~/camera_info", 1)

        # Create service for camera calibration
        # self.srv_set_camera_info = self.create_service(SetCameraInfo,
        #     "~/set_camera_info", self.srv_set_camera_info_cb)

        # monitor
        self._last_image_published_time = 0
        # ---

    @property
    def is_stopped(self):
        return self._is_stopped

    def publish(self, image_msg):
        stamp = self.get_clock().now().to_msg()
        image_msg.header.stamp = stamp
        self.current_camera_info.header.stamp = stamp
        # update camera frame
        image_msg.header.frame_id = self._frame_id
        # publish image
        self.pub_img.publish(image_msg)
        # publish camera info
        self.pub_camera_info.publish(self.current_camera_info)
        self._last_image_published_time = time.time()
        if not self._has_published:
            self.get_logger().info("Published first image.")
            self._has_published = True

    def start(self):
        """Begins the camera capturing"""
        self.get_logger().info("Start capturing.")
        try:
            try:
                self.setup()
            except RuntimeError as e:
                self.get_logger().fatal(f"{e}")
                raise e
            # run camera thread
            self._worker = Thread(target=self.run, daemon=True)
            self._worker.start()
        except StopIteration as e:
            self.get_logger().error(f"{e}")

    def stop(self, force: bool=False):
        self.get_logger().info("Stopping camera...")
        self._is_stopped = True
        if not force:
            # wait for the camera thread to finish
            if self._worker is not None:
                self._worker.join()
                time.sleep(1)
        self._worker = None
        # release resources
        self.release(force=force)
        time.sleep(1)
        self._is_stopped = False
        self.get_logger().info('Camera stopped.')

    @abstractmethod
    def setup(self):
        raise NotImplementedError('Child classes should implement this method.')

    @abstractmethod
    def release(self, force: bool = False):
        raise NotImplementedError('Child classes should implement this method.')

    @abstractmethod
    def run(self):
        raise NotImplementedError('Child classes should implement this method.')

    def on_shutdown(self):
        self.stop(force=True)

    def update_camera_params(self):
        """ Update the camera parameters based on the current resolution.
        The camera matrix, rectification matrix, and projection matrix depend on
        the resolution of the image.
        As the calibration has been done at a specific resolution, these matrices need
        to be adjusted if a different resolution is being used.
        """
        scale_width = float(self._res_w) / self.original_camera_info.width
        scale_height = float(self._res_h) / self.original_camera_info.height

        scale_matrix = np.ones(9)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[4] *= scale_height
        scale_matrix[5] *= scale_height

        # adjust the camera matrix resolution
        self.current_camera_info.height = self._res_h
        self.current_camera_info.width = self._res_w

        # adjust the K matrix
        self.current_camera_info.K = np.array(self.original_camera_info.K) * scale_matrix

        # adjust the P matrix
        scale_matrix = np.ones(12)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[5] *= scale_height
        scale_matrix[6] *= scale_height
        self.current_camera_info.P = np.array(self.original_camera_info.P) * scale_matrix

    def _get_parameters(self, params: Optional[List[DeclareParams]]):
        self.declare_parameter("framerate")
        self.declare_parameter("res_w")
        self.declare_parameter("res_h")
        self.declare_parameter("exposure_mode")
        self.declare_parameter("veh")

        if params is not None:
            for param in params:
                if param.default_value is None:
                    self.declare_parameter(param.name)
                else:
                    self.declare_parameter(param.name, param.default_value)
                # This is a hacky way to get the parameter value out without
                # knowing beforehand the parameter type
                name = f"{param.type}".split(".")[-1].lower() + "_value"
                val = getattr(
                    self.get_parameter(param.name).get_parameter_value(), name)
                setattr(self, f"_{param.name}", val)

        self._framerate = self.get_parameter("framerate").get_parameter_value()\
            .integer_value
        self._res_w = self.get_parameter("res_w").get_parameter_value()\
            .integer_value
        self._res_h = self.get_parameter("res_h").get_parameter_value()\
            .integer_value
        self._exposure_mode = self.get_parameter("exposure_mode")\
            .get_parameter_value().string_value
        self._veh = self.get_parameter("veh").get_parameter_value().string_value

    def _params_callback(self, params):
        """Callback method when parameters are updated.
        If parameters updated successfully, stop and camera to allow the
        updated parameters to take effect.
        """
        success = False
        for param in params:
            if param.name == "framerate" and param.type == Parameter.Type.INTEGER:
                success = True
                self._framerate = param.value
            elif param.name == "res_w" and param.type == Parameter.Type.INTEGER:
                success = True
                self._res_w = param.value
            elif param.name == "res_h" and param.type == Parameter.Type.INTEGER:
                success = True
                self._res_h = param.value
            elif param.name == "exposure_mode" and param.type == Parameter.Type.STRING:
                success = True
                self._exposure_mode = param.value

        if success:
            self.stop()
            self.update_camera_params()
            self.start()

        return SetParametersResult(successful=success)

    @staticmethod
    def load_camera_info(rel_file_path: str):
        """Loads the camera calibration files.
        Loads the intrinsic and extrinsic camera matrices.
        Args:
            filename (:obj:`str`): filename of calibration files.
        Returns:
            :obj:`CameraInfo`: a CameraInfo message object
        """
        calib_data, full_fp = calibration_file.read_calibration(
            rel_file_path)

        if calib_data is None:
            return None, full_fp

        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']

        # Ensure that array of elements are floating point
        to_floats = lambda l : list([float(e) for e in l])
        calib_data['camera_matrix']['data'] = to_floats(
            calib_data['camera_matrix']['data'])
        calib_data['distortion_coefficients']['data'] = to_floats(
            calib_data['distortion_coefficients']['data'])
        calib_data['rectification_matrix']['data'] = to_floats(
            calib_data['rectification_matrix']['data'])
        calib_data['projection_matrix']['data'] = to_floats(
            calib_data['projection_matrix']['data'])

        try:
            cam_info.k = calib_data['camera_matrix']['data']
            cam_info.d = calib_data['distortion_coefficients']['data']
            cam_info.r = calib_data['rectification_matrix']['data']
            cam_info.p = calib_data['projection_matrix']['data']
        except AttributeError as _:
            cam_info.K = calib_data['camera_matrix']['data']
            cam_info.D = calib_data['distortion_coefficients']['data']
            cam_info.R = calib_data['rectification_matrix']['data']
            cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        return cam_info, full_fp

    def _load_calibration_file(self):
        self.rel_cali_file = os.path.join(
            AbsCameraNode.CALIBRATION_RELATIVE_FOLDER, f"{self._veh}.yaml")
        self.original_camera_info, fp = self.load_camera_info(self.rel_cali_file)
        # If cannot find load calibration file, then try to load the default
        if self.original_camera_info is None:
            self.get_logger().warn(f"calibration file {fp}"\
            " not found. Attempting to load default calibration file instead")

            self.rel_cali_file = os.path.join(
                AbsCameraNode.CALIBRATION_RELATIVE_FOLDER, "default.yaml")
            self.original_camera_info, fp = self.load_camera_info(self.rel_cali_file)
            if self.original_camera_info is None:
                self.get_logger().fatal(f"Cannot find default calibration file"\
                    f": {fp}")
                raise CalibrationFileNotFoundException(f"{fp}"\
                    " not found")
