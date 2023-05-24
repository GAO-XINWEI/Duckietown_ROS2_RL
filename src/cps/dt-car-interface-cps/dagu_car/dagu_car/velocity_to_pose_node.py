#!/usr/bin/env python3

import math
import rclpy

from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

from dt_interfaces_cps.msg import (
    Pose2DStamped,
    Twist2DStamped
)


class VelocityToPoseNode(Node):
    """VelocityToPoseNode integrates the velocity of the Duckiebot in order to
    continuously obtain a pose relative to the pose at which the node
    was started.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node
            that ROS will use
    Subscriber:
        ~velocity (:obj:`Twist2DStamped`): The robot velocity, typically
            obtained from forward kinematics
    Publisher:
        ~pose (:obj:`Pose2DStamped`): The integrated pose relative to the pose
            of the robot at node initialization
    """
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.load_launch_parameters()

        # Keep track of the last known pose
        self.last_pose = Pose2DStamped(
            header=Header(stamp=Time(sec=0, nanosec=0)))
        self.last_theta_dot = 0
        self.last_v = 0

        self.pub_pose = self.create_publisher(Pose2DStamped, "~/pose", 1)

        self.sub_velocity = self.create_subscription(
            Twist2DStamped, "~/velocity", self.velocity_callback, 10)

    def load_launch_parameters(self):
        self.declare_parameter("veh")
        self.veh_name = self.get_parameter("veh")\
            .get_parameter_value().string_value

    def velocity_callback(self, msg_velocity: Twist2DStamped):
        """Performs the calclulation from velocity to pose and publishes a
            messsage with the result.
        Args:
            msg_velocity (:obj:`Twist2DStamped`): the current velocity message
        """
        # Skip the first frame
        if self.last_pose.header.stamp.sec > 0:
            dt = (msg_velocity.header.stamp.sec
                - self.last_pose.header.stamp.sec)

            # Integrate the relative movement between the last pose
            # and the current
            theta_delta = self.last_theta_dot * dt
            # to ensure no division by zero for radius calculation:
            if abs(self.last_theta_dot) < 0.000001:
                # straight line
                x_delta = self.last_v * dt
                y_delta = 0
            else:
                # arc of circle
                radius = self.last_v / self.last_theta_dot
                x_delta = radius * math.sin(theta_delta)
                y_delta = radius * (1.0 - math.cos(theta_delta))

            # Add to the previous to get absolute pose relative to
            # the starting position
            theta_res = self.last_pose.theta + theta_delta
            x_res = (
                self.last_pose.x
                + x_delta * math.cos(self.last_pose.theta)
                - y_delta * math.sin(self.last_pose.theta)
            )
            y_res = (
                self.last_pose.y
                + y_delta * math.cos(self.last_pose.theta)
                + x_delta * math.sin(self.last_pose.theta)
            )

            # Update the stored last pose
            self.last_pose.theta = theta_res
            self.last_pose.x = x_res
            self.last_pose.y = y_res

            # Stuff the new pose into a message and publish
            msg_pose = Pose2DStamped()
            msg_pose.header = msg_velocity.header
            msg_pose.header.frame_id = self.veh_name
            msg_pose.theta = theta_res
            msg_pose.x = x_res
            msg_pose.y = y_res
            self.pub_pose.publish(msg_pose)

        self.last_pose.header.stamp = msg_velocity.header.stamp
        self.last_theta_dot = msg_velocity.omega
        self.last_v = msg_velocity.v


def main(args=None):
    rclpy.init(args=args)
    node = VelocityToPoseNode("velocity_to_pose_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
