# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import yaml
import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo


class VideoStreamerNode(Node):
    """Main ROS Camera simulator Node function. Takes input from USB webcam
    and publishes a ROS CompressedImage and Image message to topics.
    """
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.load_launch_parameters()
        config = self.load_config_file(self.config_file_path)
        if config is not None:
            self.camera_info = self.get_camera_info(config)
        else:
            self.camera_info = None

        self.bridge = CvBridge()

        # Publishers
        self.image_publisher_ = self.create_publisher(
            CompressedImage,
            self.image_topic_name,
            5)
        self.camera_info_publisher_ = self.create_publisher(
            CameraInfo,
            self.info_topic_name,
            1)

        if self.type == "video":
            if not os.path.isfile(self.path):
                raise RuntimeError(f"Invalid video path: {self.path}")
            try:
                self.vc = cv2.VideoCapture(self.path)
                self.vc.set(cv2.CAP_PROP_POS_MSEC, self.start)
            except:
                print("End of file")
            video_fps = self.vc.get(cv2.CAP_PROP_FPS)
        elif self.type == "image":
            if not os.path.isfile(self.path):
                raise RuntimeError(f"Invalid image path: {self.path}")
            self.image = cv2.imread(self.path)
            video_fps = 10
        else:
            raise ValueError(f"Unknown type: {self.type}")

        self.timer = self.create_timer(1.0/video_fps, self.image_callback)
        self.get_logger().info(f"Publishing image at {video_fps} fps")

    def load_launch_parameters(self):
        """Load the launch ROS parameters
        """
        self.declare_parameter("config_file_path")
        self.declare_parameter("image_topic_name")
        self.declare_parameter("info_topic_name")
        self.declare_parameter("loop")
        self.declare_parameter("frame_id")
        self.declare_parameter("type")
        self.declare_parameter("path")
        self.declare_parameter("start")

        self.config_file_path = self.get_parameter("config_file_path")\
            .get_parameter_value().string_value
        
        image_topic_name = self.get_parameter("image_topic_name")\
            .get_parameter_value().string_value
        self.image_topic_name = "~/image/compressed" if image_topic_name == "" \
            else image_topic_name

        info_topic_name = self.get_parameter("info_topic_name")\
            .get_parameter_value().string_value
        self.info_topic_name = "~/camera_info" if info_topic_name == "" \
            else info_topic_name

        self.loop = self.get_parameter("loop")\
            .get_parameter_value().bool_value
        self.frame_id_ = self.get_parameter("frame_id")\
            .get_parameter_value().string_value
        self.type = self.get_parameter("type")\
            .get_parameter_value().string_value
        self.path = self.get_parameter("path")\
            .get_parameter_value().string_value
        self.start = self.get_parameter("start")\
            .get_parameter_value().integer_value

    def load_config_file(self, file_path: str):
        try:
            f = open(file_path)
            return yaml.safe_load(f)
        except IOError:
            self.get_logger().warning(
                "Could not find calibration file " + file_path +
                ", will proceed without a calibration file")
            return None

    def get_camera_info(self, config):
        ci = CameraInfo()
        ci.header.frame_id = self.frame_id_
        ci.width = config["image_width"]
        ci.height = config["image_height"]
        ci.distortion_model = config["distortion_model"]
        ci.d = list(float(v) for v in config["distortion_coefficients"]["data"])
        ci.k = list(float(v) for v in config["camera_matrix"]["data"])
        ci.r = list(float(v) for v in config["rectification_matrix"]["data"])
        ci.p = list(float(v) for v in config["projection_matrix"]["data"])
        return ci

    def image_callback(self):
        if self.type == "video":
            rval, image = self.vc.read()
            if not rval and not self.loop:
                self.get_logger().info("End of video, closing node...")
                self.timer.cancel()
                self.destroy_node()
                exit()
            elif not rval and self.loop:
                self.vc.set(cv2.CAP_PROP_POS_MSEC, 0)
                rval, image = self.vc.read()
        elif self.type == "image":
            image = self.image

        time_msg = self.get_clock().now().to_msg()
        img_msg = self.get_image_msg(image, time_msg)

        if self.camera_info is not None:
            self.camera_info.header.stamp = time_msg
            self.camera_info_publisher_.publish(self.camera_info)

        self.image_publisher_.publish(img_msg)

    def get_image_msg(self, image, time):
        """Get image message, takes image as input and returns CvBridge
        image message
        :param image: cv2 image
        :return: sensor_msgs/Imag
        """
        img_msg = self.bridge.cv2_to_compressed_imgmsg(image)
        img_msg.header.stamp = time
        return img_msg


def main(args=None):
    rclpy.init(args=args)

    video_streamer_node = VideoStreamerNode("video_streamer_node")

    rclpy.spin(video_streamer_node)

    video_streamer_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
