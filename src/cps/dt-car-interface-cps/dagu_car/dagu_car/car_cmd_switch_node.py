#!/usr/bin/env python3

import yaml
import rclpy

from rclpy.node import Node
from functools import partial

from dt_interfaces_cps.msg import (
    FSMState,
    Twist2DStamped
)


class CarCmdSwitchNode(Node):
    """
    Subscriber:
        ~mode (:obj:`duckietown_msgs/FSMState`): Current control mode
            of the duckiebot
    Publisher:
        ~cmd (:obj`duckietown_msgs/Twist2DStamped`): The car command messages
            corresponding to the selected mode
    """
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.load_launch_parameters()
        self.load_param_file(self._param_file_path)

        self.pub_cmd = self.create_publisher(Twist2DStamped, "~/cmd", 1)

        self.sub_fsm_state = self.create_subscription(
            FSMState, self._mode_topic, self.fsm_state_cb, 1)

        self.sub_dict = {
            src_name : self.create_subscription(
                Twist2DStamped,
                topic_name,
                partial(self.wheels_cmd_cb, src_name=str(src_name)),
                1
            )
            for src_name, topic_name in self._source_topics.items()
        }

    def load_launch_parameters(self) -> str:
        self.declare_parameter("param_file_path")
        self.declare_parameter("default_src_name")
        self._param_file_path = self.get_parameter("param_file_path")\
            .get_parameter_value().string_value
        self.current_src_name = self.get_parameter("default_src_name")\
            .get_parameter_value().string_value

    def load_param_file(self, filepath: str):
        """Load YAML parameter file"""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            self._mode_topic = config['mode_topic']
            self._source_topics = config['source_topics']
            self._mappings = config['mappings']

    def fsm_state_cb(self, fsm_state_msg: FSMState):
        self.current_src_name = self._mappings.get(fsm_state_msg.state)
        if self.current_src_name == "stop":
            self.publish_stop()
            self.get_logger().info(
                f"Car cmd switched to STOP in state {fsm_state_msg.state}.")
        elif self.current_src_name is None:
            self.get_logger().warn(f"FSMState {fsm_state_msg.state} not handled."\
                " No msg pass through the switch.")
        else:
            self.get_logger().info(f"Car cmd switched to {self.current_src_name}"
                    f" in state {fsm_state_msg.state}.")

    def wheels_cmd_cb(self, msg: Twist2DStamped, src_name: str):
        if src_name == self.current_src_name:
            self.pub_cmd.publish(msg)

    def publish_stop(self):
        msg = Twist2DStamped()
        msg.v = 0
        msg.omega = 0
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CarCmdSwitchNode("car_cmd_switch_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
