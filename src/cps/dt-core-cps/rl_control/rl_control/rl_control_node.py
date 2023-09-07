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
        # Check the cuda
        self.get_logger().info(f'cuda available: {torch.cuda.is_available()}')

        # Load parameters
        self.params = {}
        self.declare_parameter('veh')
        self.declare_parameter('model_path')
        self.declare_parameter('model')
        self.declare_parameter('quantized_model')
        self.declare_parameter('device')
        self.declare_parameter('quantize')
        self.declare_parameter('discrete')
        self.params["veh"] = self.get_parameter("veh").get_parameter_value().string_value
        self.params["model_path"] = self.get_parameter("model_path").get_parameter_value().string_value
        self.params["model"] = self.get_parameter("model").get_parameter_value().string_value
        self.params["quantized_model"] = self.get_parameter("quantized_model").get_parameter_value().string_value
        self.params["device"] = self.get_parameter("device").get_parameter_value().string_value
        self.params["quantize"] = self.get_parameter("quantize").get_parameter_value().string_value
        self.params["discrete"] = self.get_parameter("discrete").get_parameter_value().string_value
        self.get_logger().info(f'Load parameters: device: {self.params["device"]}; quantize: {self.params["quantize"]}; discrete: {self.params["discrete"]}.')
        # Load Network
        if self.params["device"] == 'cpu' or self.params["quantize"] == 'T' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() and self.params["device"] is not 'cpu' and not self.params["bool_quantized"] else 'cpu')
        self.get_logger().info(f'Try to load with device: {self.device}')

        if self.params["quantize"] == 'T':
            self.get_logger().info(f'quantized_model')
            self.model = torch.load(self.params["model_path"] + 'quantized_model.pt', map_location=self.device)
        elif self.params["discrete"] == 'F':
            from rl_control.ACNet_continuous import ACNet
            checkpoint = torch.load(self.params["model_path"] + self.params["model"], map_location=self.device)
            self.model = ACNet(ACTION_SIZE, self.device).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            from rl_control.ACNet_discrete import ACNet
            checkpoint = torch.load(self.params["model_path"] + self.params["model"], map_location=self.device)
            self.model = ACNet(ACTION_SIZE, self.device).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        self.get_logger().info(f'Network loaded. Using device: {self.device}')

        # todo: 10 times input
        self.get_logger().info(f'Ready to input.')
        self.input = torch.tensor(255 * np.random.random(size=(1, 3, 30, 40)), dtype=torch.float32).to(self.device)
        for i in range(10):
            with torch.no_grad():
                _, _, _, _, _ = self.model(self.input, lstm_state=None, old_action=None)
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
        try:
            frame = np.fromstring(bytes(data.data), np.uint8)
            self.get_logger().info(f'point 2: new frame')
        except AttributeError:
            self.get_logger().info('Camera node is not yet ready...')

        # Preprocess
        preprocessed_img = self.preprocess_img(frame)
        self.get_logger().info(f'point 3: after preprocess')

        # Infer Action
        if self.params["discrete"] == 'F':
            action = self.infer_action_continuous(preprocessed_img)
        else:
            action = self.infer_action_discrete(preprocessed_img)
        self.get_logger().info(f'point 4: action infered')
                
        # Publish Message
        self.publish_control_msg(action)
        self.get_logger().info('point 5: Action published')

    
    def trans_state(self, obs):
        return 255 * resize_to_tensor(obs).type(torch.float32).unsqueeze(0)
    
    def preprocess_img(self, frame):
        # transfer and input
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        states = self.trans_state(frame)
        return states

    def infer_action_continuous(self, states):
        with torch.no_grad():
            actions, _, _, _, _ = self.model(states.to(self.device), lstm_state=None, old_action=None)
        action = actions.cpu().data.numpy().flatten()
        action = np.clip(action, -1, 1)
        # self.get_logger().info(f'infer action is: {action}')
        vel, angle = action
        return [vel, angle]
    
    def infer_action_discrete(self, states):
        with torch.no_grad():
            actions, _, _, _, _ = self.model(states.to(self.device), lstm_state=None, old_action=None)
        action = actions.cpu().data.numpy().flatten()
        def return_vel_steer(u_r,u_l):
            GAIN = 1.0
            TRIM = 0.0
            RADIUS = 0.0318
            K = 27.0
            LIMIT = 1.0 
            WHEEL_DIST = 0.102
            k_r = K
            k_l = K
            k_r_inv = (GAIN + TRIM)/k_r
            k_l_inv = (GAIN - TRIM)/k_l
            w_r  = u_r/k_r_inv
            w_l  = u_l/k_l_inv
            b    = WHEEL_DIST
            vel   = (w_r + w_l)*RADIUS/2
            angle = (w_r - w_l)*RADIUS/b
            return np.array([vel,angle])
        if action[0] == 0:
            action = return_vel_steer(0.4,0.04)
        if action[0] == 1:
            action = return_vel_steer(0.04,0.4)
        if action[0] == 2:
            action = return_vel_steer(0.3,0.3)
        action = np.clip(action, -1, 1)
        # self.get_logger().info(f'infer action is: {action}')
        vel, angle = action
        return [vel, angle]
        
    def publish_control_msg(self, action):
        """Publishes the output of the model as a control message."""
        vel, angle = action
        vel_left, vel_right = ActionFloat32(), ActionFloat32()
        vel_left.data, vel_right.data = [float(vel)], [float(angle)]
        msg = ActionFloat32Array2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = [vel_left, vel_right]
        self.pub_action_net.publish(msg)

        # transfer to wheel cmd
        def action_to_wheel(action):
            # Soc 1
            SPEED_FACTOR = 0.8
            ANGLE_FACTOR = 4
            # Soc 2
            # SPEED_FACTOR = 0.6
            # ANGLE_FACTOR = 4

            vel, angle = action
            # Distance between the wheels
            # assuming same motor constants k for both motors
            # adjusting k by gain and trim
            k_r_inv = (GAIN + TRIM) / K
            k_l_inv = (GAIN - TRIM) / K
            omega_r = (vel * SPEED_FACTOR + 0.5 * angle * WHEEL_DIST * ANGLE_FACTOR) / RADIUS
            omega_l = (vel * SPEED_FACTOR - 0.5 * angle * WHEEL_DIST * ANGLE_FACTOR) / RADIUS
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

