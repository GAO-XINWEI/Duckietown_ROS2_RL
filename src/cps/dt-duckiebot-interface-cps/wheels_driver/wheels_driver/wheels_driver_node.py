#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from dt_interfaces_cps.msg import WheelsCmdStamped, BoolStamped
from wheels_driver.dagu_wheels_driver import DaguWheelsDriver


class WheelsDriverNode(Node):
    """Node handling the motor velocities communication.
        Subscribes to the requested wheels commands (linear velocities, i.e.
        velocity for the left and the right wheels) and to an
        emergency stop flag.
        When the emergency flag `~emergency_stop` is set to `False` it actuates
        the wheel drive with the velocities received from `~wheels_cmd`.
        Publishes the execution of the commands to `~wheels_cmd_executed`.
        The emergency flag is `False` by default.
        Subscribers:
           ~wheels_cmd (:obj:`WheelsCmdStamped`): The requested wheel command
           ~emergency_stop (:obj:`BoolStamped`): Emergency stop. Can stop
                the actual execution of
                the wheel commands by the motors if set to `True`. Set to
                `False` for nominal operations.
        Publishers:
           ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Publishes the
                actual commands executed, i.e. when the emergency flag is
                `False` it publishes the requested command, and when it is
                `True`: zero values for both motors.
    """    
    def __init__(self, node_name: str):
        super().__init__(node_name)
        
        self.estop = False
        
        # Setup the driver
        self.driver = DaguWheelsDriver()

        # Initialize the executed commands message
        self.msg_wheels_cmd = WheelsCmdStamped()

        # Publisher for wheels command wih execution time
        self.pub_wheels_cmd = self.create_publisher(
            WheelsCmdStamped,
            "~/wheels_cmd_executed",
            1)

        # Subscribers
        self.sub_topic = self.create_subscription(
            WheelsCmdStamped,
            "~/wheels_cmd",
            self.wheels_cmd_cb,
            1)
        self.sub_e_stop = self.create_subscription(
            BoolStamped,
            "~/emergency_stop",
            self.estop_cb,
            1)

        self.get_logger().info("Initialised")

    def wheels_cmd_cb(self, msg: WheelsCmdStamped):
        """
        Callback that sets wheels' speeds.
            Creates the wheels' speed message and publishes it. If the
            emergency stop flag is activated, publishes zero command.
            Args:
                msg (WheelsCmdStamped): velocity command
        """
        if self.estop:
            vel_left = 0.0
            vel_right = 0.0
        else:
            vel_left = msg.vel_left
            vel_right = msg.vel_right

        self.driver.set_wheels_speed(left=vel_left, right=vel_right)
        # Put the wheel commands in a message and publish
        self.msg_wheels_cmd.header = msg.header
        # Record the time the command was given to the wheels_driver
        self.msg_wheels_cmd.header.stamp = self.get_clock().now().to_msg()
        self.msg_wheels_cmd.vel_left = vel_left
        self.msg_wheels_cmd.vel_right = vel_right
        self.pub_wheels_cmd.publish(self.msg_wheels_cmd)

    def estop_cb(self, msg: BoolStamped):
        """
        Callback that enables/disables emergency stop
            Args:
                msg (BoolStamped): emergency_stop flag
        """
        self.estop = msg.data
        if self.estop:
            self.get_logger().info("Emergency Stop Activated!!!")
        else:
            self.get_logger().info("Emergency Stop Released!!!")
    
    def on_shutdown(self):
        """Shutdown procedure.
        Publishes a zero velocity command at shutdown.
        """
        self.driver.set_wheels_speed(left=0.0, right=0.0)


def main(args=None):
    rclpy.init(args=args)
    node = WheelsDriverNode(node_name="wheels_driver_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
