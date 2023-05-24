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
from dt_interfaces_cps.msg import WheelsCmd
from dt_rl_interfaces_cps.msg import ActionFloat32Array2

resize_to_tensor = tran.Compose([tran.ToPILImage(),
                                 tran.Resize(STATE_SIZE),
                                 tran.ToTensor()])

class RLControl(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        # CV Bridge
        self.bridge = CvBridge()
        self.lock = Lock()
        self.last_frame = None
        
        # Load parameters
        params_file = os.path.join(os.path.dirname(__file__), 'policy_params.yml')
        self.params = {}
        with open(params_file, 'r') as paramf:
            self.params = yaml.safe_load(paramf.read())
        rospy.loginfo('Loaded the following parameters:')
        for param in self.params:
            rospy.loginfo(f'{param}: {self.params[param]}')

        # Load Network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {device}')
        checkpoint = torch.load(os.path.dirname(__file__) + "/" + self.params["model"], map_location=self.device)
        self.model = ACNet(ACTION_SIZE, self.device).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        #Additional Info when using cuda
        if self.device.type == 'cuda':
            self.get_logger().info(torch.cuda.get_device_name(0))
            self.get_logger().info('Memory Usage:')
            self.get_logger().info(f'Allocated: \
{round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
            self.get_logger().info(f'Cached: \
{round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')

        # Subscribers
        self.sub_img = self.create_subscription(
            CompressedImage,
            "~/image/compressed",
            self.img_callback,
            1)

        # Publishers
        self.pub_action_net = self.create_publisher(
            Action_Float32Array2,
            "~/rl/action_net",
            1)

        self.get_logger().info("Initialized")
        
        
    def img_callback(self, data):
        # rospy.loginfo(f'Image Callback')
        if self.lock.acquire(True, timeout=0.05):
            try:
                self.last_frame = data
            finally:
                self.lock.release()
    
    def trans_state(self, obs):
        return 255 * resize_to_tensor(obs).type(torch.float32).unsqueeze(0)
    
    def preprocess_img(self, frame):
        # transfer and input
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        states = self.trans_state(frame)
        return states

    def infer_action(self, states):
        lstm_states = None
        with torch.no_grad():
            actions, _, lstm_states, _, _ = self.model(states.to(self.device), lstm_state=lstm_states, old_action=None)
        action = actions.cpu().data.numpy().flatten()
        vel, angle = action
        return [vel, angle]
        
    def publish_control_msg(self, action):
        """Publishes the output of the model as a control message."""
        vel, angle = action
        vel_left, vel_right = vel, angle
        # msg = WheelsCmd()
        # msg.vel_left = vel_left
        # msg.vel_right = vel_right
        msg = ActionFloat32Array2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = action
        self.pub_action_net.publish(msg)
        
    def process_action(self, states):
        while rclpy.ok():
            if self.last_frame is not None:
                self.lock.acquire()
                try:
                    frame = np.fromstring(self.last_frame.data, np.uint8)
                    self.last_frame = None
                except AttributeError:
                    rospy.logwarn('Camera node is not yet ready...')
                    continue
                finally:
                    self.lock.release()

                # Preprocess
                preprocessed_img = self.preprocess_img(frame)

                # Infer Action
                action = self.infer_action(preprocessed_img)
                
                # Publish Message
                self.publish_control_msg(action)

                self.get_logger().info('Action published')


def main(args=None):
    rclpy.init(args=args)
    node = RLControl("rl_control_node")
    try:
        # run the model in a separate thread
        thread = Thread(target=node.process_action)
        thread.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

