#!/usb/bin/env python3

# Test the `velocity_to_pose` node by publishing to the topics it subscribes to.

import rclpy

from rclpy.node import Node
from std_msgs.msg import Header

from dt_interfaces_cps.msg import (
    Twist2DStamped
)


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('test_vtp_node')
        self.publisher_ = self.create_publisher(
            Twist2DStamped, 'velocity', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Twist2DStamped()
        msg.header = Header()
        msg.header.frame_id = str(self.i)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.v = 0.1 * self.i
        msg.omega = 1.0 * self.i
        self.publisher_.publish(msg)
        self.i = (self.i + 1) % 10


def main(args=None):
    rclpy.init(args=args)
    node = MinimalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
