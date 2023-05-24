import rclpy

from rclpy.node import Node
from std_msgs.msg import Header

from dt_interfaces_cps.msg import Twist2DStamped


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('test_kinematics_node')
        self.publisher_ = self.create_publisher(
            Twist2DStamped, 'car_cmd', 10)
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

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
