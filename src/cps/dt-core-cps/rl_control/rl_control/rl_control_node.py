#!/usr/bin/env python3

import os
import re
import cv2
import sys
import rclpy
import pprint
import numpy as np

import torch
import torchvision
import torchvision.transforms as tran

from typing import List
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage


from threading import Event, Lock, Thread
from rl_control.ACNet import ACNet
from rl_control.Parameters import *
from dt_interfaces_cps.msg import WheelsCmdStamped
from dt_rl_interfaces_cps.msg import ActionFloat32Array2, ActionFloat32

resize_to_tensor = tran.Compose([tran.ToPILImage(),
                                 tran.Resize(STATE_SIZE),
                                 tran.ToTensor()])

class RLControl(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        # CV Bridge
        self.bridge = CvBridge()
        self.last_frame = None
        
        # Load parameters
        self.params = {}
        self.declare_parameter('veh')
        self.declare_parameter('model')
        self.declare_parameter('device')
        self.declare_parameter('model_path')
        self.params["veh"] = self.get_parameter("veh").get_parameter_value().string_value
        self.params["model"] = self.get_parameter("model").get_parameter_value().string_value
        self.params["device"] = self.get_parameter("device").get_parameter_value().string_value
        self.params["model_path"] = self.get_parameter("model_path").get_parameter_value().string_value
        self.get_logger().info(f'Load parameters: device: {self.params["device"]}')

        # Load Network
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Try to load with device: {self.device}')
        checkpoint = torch.load(self.params["model_path"] + "/" + self.params["model"], map_location=self.device)
        self.model = ACNet(ACTION_SIZE, self.device).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.get_logger().info(f'Network loaded. Using device: {self.device}')
        # Additional Info when using cuda
        if self.device.type == 'cuda':
            self.get_logger().info(torch.cuda.get_device_name(0))

        # todo: 10 times input
        self.get_logger().info(f'Ready to input.')
        test_tensor = 255 * np.random.random(size=(1, 3, 30, 40))
        for i in range(10):
            with torch.no_grad():
                _, _, _, _, _ = self.model(torch.tensor(test_tensor, dtype=torch.float32).to(self.device), lstm_state=None, old_action=None)
            self.get_logger().info(f'Run {i} times input done!')

        # Subscribers
        self.sub_img = self.create_subscription(
            CompressedImage,
            "camera_node/image/compressed",
            self._img_callback,
            1)

        # Publishers
        self.pub_action_net = self.create_publisher(
            ActionFloat32Array2,
            "rl/action_cmd",
            1)
        self.pub_action_motor = self.create_publisher(
            WheelsCmdStamped,
            "wheels_driver_node/wheels_cmd",
            1)

        self.get_logger().info("Initialized")

    def _img_callback(self, data):
        self.get_logger().info(f'point 1: entry')
        # self.get_logger().info(f'point 2: lock')
        try:
            frame = np.fromstring(bytes(data.data), np.uint8)
            self.get_logger().info(f'point 3: new frame')
        except AttributeError:
            self.get_logger().info('Camera node is not yet ready...')

        # Preprocess
        # self.get_logger().info(f'point 4: pre preprocess')
        preprocessed_img = self.preprocess_img(frame)
        self.get_logger().info(f'point 5: after preprocess')

        # Infer Action
        action = self.infer_action(preprocessed_img)
        self.get_logger().info(f'point 6: action infered')
                
        # Publish Message
        self.publish_control_msg(action)
        self.get_logger().info('Action published')

    
    def trans_state(self, obs):
        return 255 * resize_to_tensor(obs).type(torch.float32).unsqueeze(0)
    
    def preprocess_img(self, frame):
        # transfer and input
        self.get_logger().info(f'preprocess_img()')
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        states = self.trans_state(frame)
        return states

    def infer_action(self, states):
        lstm_states = None
        with torch.no_grad():
            actions, _, lstm_states, _, _ = self.model(states.to(self.device), lstm_state=lstm_states, old_action=None)
        action = actions.cpu().data.numpy().flatten()
        action = np.clip(action, -1, 1)
        self.get_logger().info(f'infer action is: {action}')
        vel, angle = action
        return [vel, angle]
        
    def publish_control_msg(self, action):
        """Publishes the output of the model as a control message."""
        self.get_logger().info(f'publish_control_msg()')

        vel, angle = action
        vel_left, vel_right = ActionFloat32(), ActionFloat32()
        vel_left.data, vel_right.data = [float(vel)], [float(angle)]
        msg = ActionFloat32Array2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = [vel_left, vel_right]
        self.pub_action_net.publish(msg)

        # transfer to wheel cmd
        def action_to_wheel(action):
            vel, angle = action
            # Distance between the wheels
            # assuming same motor constants k for both motors
            # adjusting k by gain and trim
            k_r_inv = (GAIN + TRIM) / K
            k_l_inv = (GAIN - TRIM) / K
            omega_r = (vel + 0.5 * angle * WHEEL_DIST) / RADIUS
            omega_l = (vel - 0.5 * angle * WHEEL_DIST) / RADIUS
            # conversion from motor rotation rate to duty cycle
            u_r = omega_r * k_r_inv
            u_l = omega_l * k_l_inv
            # limiting output to limit, which is 1.0 for the duckiebot
            u_r_limited = max(min(u_r, LIMIT), -LIMIT)
            u_l_limited = max(min(u_l, LIMIT), -LIMIT)
            return u_l_limited, u_r_limited

        vel_left, vel_right = action_to_wheel(action)
        msg = WheelsCmdStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vel_left = float(vel_left)
        msg.vel_right = float(vel_right)
        self.pub_action_motor.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = RLControl("rl_control_node")
    try:
        print('rl_control_node > main(): Spining...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

