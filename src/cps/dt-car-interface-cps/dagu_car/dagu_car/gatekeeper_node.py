#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from dt_ood_interfaces_cps.msg import OODAlert
from dt_interfaces_cps.msg import BoolStamped, Twist2DStamped


class GatekeeperNode(Node):
    """Gatekeeps the `car_cmd` before being sent to the `kinematics_node`.
    Either stops or passthrough the `car_cmd` messages.
    """
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.get_launch_params()

        self.ood_alert = False

        # Subscribers
        self.sub_car_cmds = self.create_subscription(
            Twist2DStamped,
            "~/in_car_cmd",
            self.cb_car_cmd,
            1)
        self.sub_start = self.create_subscription(
            BoolStamped,
            "~/activate",
            self.cb_start,
            1)
        self.sub_ood_alert = self.create_subscription(
            OODAlert,
            "~/ood_alert",
            self.cb_ood_alert,
            1)

        # Publisher
        self.pub_car_cmd = self.create_publisher(
            Twist2DStamped,
            "~/out_car_cmd",
            1)

        self.get_logger().info(f"Activated with passthrough: {self.activated} \
OOD Alert: {self.ood_alert}")

        if not self.activated:
            self.get_logger().warn(
                f"Wheel commands not passthrough by default!")

    def get_launch_params(self):
        self.declare_parameter("default_passthrough")
        self.activated = self.get_parameter("default_passthrough")\
            .get_parameter_value().bool_value

    def cb_car_cmd(self, wheels_cmd: Twist2DStamped):
        """Callback for car commands."""
        if self.activated and not self.ood_alert:
            self.pub_car_cmd.publish(wheels_cmd)

    def cb_start(self, msg: BoolStamped):
        self.activated = msg.data
        if self.activated:
            self.get_logger().info(f"Gatekeeper activated")
        else:
            self.get_logger().info(f"Gatekeeper deactivated, sending \
stop command")
            self.publish_stop_command()

    def cb_ood_alert(self, msg: OODAlert):
        _alert = msg.ood

        # If transition from no OOD alert to OOD alert
        if not self.ood_alert and _alert:
            self.get_logger().warn("OOD alert triggered! Sending stop command")
            self.publish_stop_command()
        # If transition from OOD alert to no alert
        elif self.ood_alert and not _alert:
            self.get_logger().warn("OOD alert untriggered")

        self.ood_alert = _alert

    def publish_stop_command(self):
        self.pub_car_cmd.publish(Twist2DStamped(v=0.0, omega=0.0))


def main(args=None):
    rclpy.init(args=args)
    node = GatekeeperNode("gatekeeper_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
