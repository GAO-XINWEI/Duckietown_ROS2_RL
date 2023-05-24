#!/usr/bin/env python3

import math
import json
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Joy

from dt_interfaces_cps.msg import (
    Twist2DStamped,
    BoolStamped
)


class JoyMapperNode(Node):
    """Interprets the Joystick commands.
    The `JoyMapperNode` receives :obj:`Joy` messages from a physical joystick
    or a virtual one, interprets the buttons presses and acts accordingly.
    **Joystick bindings:**
    +----------------------+------------------+------------------------------------------------+
    | Physical joystick    | Virtual joystick | Action                                         |
    +======================+==================+================================================+
    | Directional controls | Arrow keys       | Move the Duckiebot (if not in lane-following)  |
    +----------------------+------------------+------------------------------------------------+
    | Start button         | `A` key          | Start lane-following                           |
    +----------------------+------------------+------------------------------------------------+
    | Back button          | `S` key          | Stop lane-following                            |
    +----------------------+------------------+------------------------------------------------+
    | Y button             | `E` key          | Toggle Emergency Stop                          |
    +----------------------+------------------+------------------------------------------------+
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that
            ROS will use
    Configuration:
        ~speed_gain (:obj:`float`): Gain for the directional joystick
            keys (forward/reverse)
        ~steer_gain (:obj:`int`): Gain for the directional joystick keys
            (steering angle)
        ~bicycle_kinematics (:obj:`bool`): `True` for bicycle kinematics;
            `False` for holonomic kinematics. Default is `False`
        ~simulated_vehicle_length (:obj:`float`): Used in bicycle kinematics
            model
    Subscriber:
        joy (:obj:`Joy`): The command read from joystick
        emergency_stop (:obj:`BoolStamped`): The emergency stop status
    Publishers:
        ~car_cmd (:obj:`duckietown_msgs/Twist2DStamped`): Wheels command for
            Duckiebot, based on the directional buttons pressed
        ~joystick_override (:obj:`duckietown_msgs/BoolStamped`): Boolean that is
            used to control whether lane-following or joystick control is on
    """
    def __init__(self, node_name: str):
        super().__init__(node_name)

        # emergency stop disabled by default
        self.e_stop = False

        self.load_launch_parameters()

        self.pub_car_cmd = self.create_publisher(Twist2DStamped, "~/car_cmd", 1)
        self.pub_joy_override = self.create_publisher(
            BoolStamped, "~/joystick_override", 1)
        self.pub_e_stop = self.create_publisher(
            BoolStamped, "~/emergency_stop", 1)

        self.sub_joy = self.create_subscription(Joy, "~/joy", self.joy_cb, 1)
        self.sub_e_stop = self.create_subscription(
            BoolStamped, "~/emergency_stop", self.estop_cb, 1)

        self.get_logger().info(f"Initialized with"
            f" {json.dumps(self.get_current_config(),sort_keys=True, indent=4)}")

    def load_launch_parameters(self):
        self.declare_parameter("speed_gain")
        self.declare_parameter("steer_gain")
        self.declare_parameter("bicycle_kinematics")
        self.declare_parameter("simulated_vehicle_length")

        self._speed_gain = self.get_parameter("speed_gain")\
            .get_parameter_value().double_value
        self._steer_gain = self.get_parameter("steer_gain")\
            .get_parameter_value().double_value
        self._bicycle_kinematics = self.get_parameter("bicycle_kinematics")\
            .get_parameter_value().bool_value
        self._simulated_vehicle_length = self.get_parameter(
            "simulated_vehicle_length").get_parameter_value().double_value

    def get_current_config(self):
        return {
            "speed_gain" : self._speed_gain,
            "steer_gain" : self._steer_gain,
            "bicycle_kinematics" : self._bicycle_kinematics,
            "simulated_vehicle_length" : self._simulated_vehicle_length
        }

    def estop_cb(self, estop_msg: BoolStamped):
        """Callback that process the received :obj:`BoolStamped` messages.
        Args:
            estop_msg (:obj:`BoolStamped`): the emergency_stop
                message to process.
        """
        self.e_stop = estop_msg.data

    def joy_cb(self, joy_msg: Joy):
        """Callback that process the received :obj:`Joy` messages.
        Args:
            joy_msg (:obj:`Joy`): the joystick message to process.
        """
        # Navigation buttons
        car_cmd_msg = Twist2DStamped()
        car_cmd_msg.header.stamp = self.get_clock().now().to_msg()
        # Left stick V-axis. Up is positive
        car_cmd_msg.v = joy_msg.axes[1] * self._speed_gain
        if self._bicycle_kinematics:
            # Implements Bicycle Kinematics - Nonholonomic Kinematics
            # see https://inst.eecs.berkeley.edu/~ee192/sp13/pdf/steer-control.pdf
            steering_angle = joy_msg.axes[3] * self._steer_gain
            car_cmd_msg.omega = (car_cmd_msg.v / self._simulated_vehicle_length
                * math.tan(steering_angle))
        else:
            # Holonomic Kinematics for Normal Driving
            car_cmd_msg.omega = joy_msg.axes[3] * self._steer_gain

        self.pub_car_cmd.publish(car_cmd_msg)

        bool_stamped_msg = BoolStamped()
        bool_stamped_msg.header.stamp = joy_msg.header.stamp

        # Back button: Stop LF
        if joy_msg.buttons[6] == 1:
            bool_stamped_msg.data = True
            self.get_logger().info("override_msg = True")
            self.pub_joy_override.publish(bool_stamped_msg)
        # Start button: Start LF
        elif joy_msg.buttons[7] == 1:
            bool_stamped_msg.data = False
            self.get_logger().info("override_msg = False")
            self.pub_joy_override.publish(bool_stamped_msg)
        # Y button: Emergency Stop
        elif joy_msg.buttons[3] == 1:
            self.e_stop = not self.e_stop
            bool_stamped_msg.data = self.e_stop
            self.pub_e_stop.publish(bool_stamped_msg)
        else:
            some_active = sum(joy_msg.buttons) > 0
            if some_active:
                self.get_logger().warn(
                    f"No binding for joy_msg.buttons = {str(joy_msg.buttons)}")


def main(args=None):
    rclpy.init(args=args)
    node = JoyMapperNode("joy_mapper")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
