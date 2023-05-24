#!/usr/bin/env python3

import json
import rclpy
import os.path

from typing import List

from rclpy.node import Node
from std_srvs.srv import Trigger
from rclpy.parameter import Parameter
from rcl_interfaces.msg import (
    ParameterType,
    FloatingPointRange,
    ParameterDescriptor,
    SetParametersResult
)

import dt_calibration_utils.calibration_file as calibration_utils

from dt_interfaces_cps.msg import (
    Twist2DStamped,
    WheelsCmdStamped
)


class KinematicsNode(Node):
    """
    The `KinematicsNode` maps car commands send from various nodes to wheel
    commands that the robot can execute.
    The `KinematicsNode` performs both the inverse and forward kinematics
    calculations. Before these were implemented in separate nodes, but due to
    their similarity and parameter sharing, they are now combined.
    `KinematicsNode` utilises the car geometry as well as a number of tunining
    and limiting parameters to calculate the wheel commands that the wheels
    should execute in order for the robot to perform the desired car commands
    (inverse kinematics). Then it uses these wheel commands in order to do an
    open-loop velocity estimation (the forward kinematics part).
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that
        ROS will use
    Configuration:
        ~gain (:obj:`float`): scaling factor applied to the desired velocity,
            default is 1.0
        ~trim (:obj:`float`): trimming factor that is typically used to offset
            differences in the behaviour of the left and right motors, it is
            recommended to use a value that results in the robot moving in a
            straight line when forward command is given, default is 0.0
        ~baseline (:obj:`float`): the distance between the two wheels of the
            robot, default is 0.1
        ~radius (:obj:`float`): radius of the wheel, default is 0.0318
        ~k (:obj:`float`): motor constant, assumed equal for both motors,
            default is 27.0
        ~limit (:obj:`float`): limits the final commands sent to the motors,
            default is 1.0
        ~v_max (:obj:`float`): limits the input velocity, default is 1.0
        ~omega_max (:obj:`float`): limits the input steering angle,
            default is 8.0
    Subscriber:
        ~car_cmd (:obj:`Twist2DStamped`): The requested car command
    Publisher:
        ~wheels_cmd (:obj:`WheelsCmdStamped`): The corresponding resulting
            wheel commands
        ~velocity (:obj:`Twist2DStamped`): The open-loop estimation of the
            robot velocity
    Service:
        ~save_calibration:
            Saves the current set of kinematics parameters (the ones in the
            Configuration section) to `kinematics/{ROBOT_NAME}.yaml` in the
            calibration directory.
    """
    REL_CALIBRATION_DIR = "kinematics"

    # List of all configuration names and their ParameterDescriptors
    CONFIGURATION_NAMES = {
        "k" : ParameterDescriptor(name="k",
            type=ParameterType.PARAMETER_DOUBLE, read_only=True),
        "gain" : ParameterDescriptor(name="gain",
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.1, to_value=10.0)]),
        "trim" : ParameterDescriptor(name="trim",
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0)]),
        "baseline" : ParameterDescriptor(name="baseline",
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.05, to_value=0.2)]),
        "radius" : ParameterDescriptor(name="radius",
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.01, to_value=0.1)]),
        "limit" : ParameterDescriptor(name="limit",
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.1, to_value=1.0)]),
        "v_max" : ParameterDescriptor(name="v_max",
            type=ParameterType.PARAMETER_DOUBLE,
            floating_point_range=[FloatingPointRange(from_value=0.01, to_value=2.0)]),
        "omega_max" : ParameterDescriptor(name="omega_max",
        type=ParameterType.PARAMETER_DOUBLE,
        floating_point_range=[FloatingPointRange(from_value=1.0, to_value=10.0)]),
    }

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.load_launch_parameters()
        self.read_params_from_calibration_file(self._veh_name)
        self.set_config_as_ros_params()

        self.pub_wheels_cmd = self.create_publisher(
            WheelsCmdStamped,
            "~/wheels_cmd",
            1)
        self.pub_velocity = self.create_publisher(
            Twist2DStamped,
            "~/velocity",
            1)
        self.sub_car_cmd = self.create_subscription(
            Twist2DStamped,
            "~/car_cmd",
            self.car_cmd_callback,
            1)

        self.srv_save_config = self.create_service(
            Trigger,
            "~/save_calibrations",
            self.save_calibration_callback)

    def load_launch_parameters(self):
        """Load launch parameters."""
        self.declare_parameter("veh")
        self._veh_name = self.get_parameter("veh").get_parameter_value()\
            .string_value

    def get_current_configuration(self):
        """Get the current configuration values being used."""
        return {config : getattr(self, f"_{config}")
            for config in self.CONFIGURATION_NAMES.keys()}

    def read_params_from_calibration_file(self, veh: str):
        """Read in calibration file.
        First try to read in calibration file from the specific to the robot.
        If unable to find or read, then use the default calibration file.
        If still unable to find default file, then raise exception.
        """
        calibration_data = None
        full_fp = None
        calibration_data, full_fp = calibration_utils.read_calibration(
            os.path.join(self.REL_CALIBRATION_DIR, f"{veh}.yaml"))

        if calibration_data is None:
            self.get_logger().info(f"Unable to find calibration data for \
{self._veh_name}, attempting to use defaults.")
            calibration_data, full_fp = calibration_utils.read_calibration(
                os.path.join(self.REL_CALIBRATION_DIR, "default.yaml"))
            # If unable to even find default calibration file, then throw
            # exception and fail
            if calibration_data is None:
                msg = f"Unable to find default calibration file."
                self.get_logger().fatal(msg)
                raise RuntimeError(msg)

        dump = json.dumps(calibration_data, sort_keys=True, indent=2)
        self.get_logger().info(f"Configuration: {dump}, from {full_fp}")

        for config_name in self.CONFIGURATION_NAMES.keys():
            setattr(self, f"_{config_name}", calibration_data[config_name])

    def set_config_as_ros_params(self):
        """Take the configuration parameters that was loaded from YAML file
        and declare them as ROS2 parameters for easier introspection and
        changing the parameters on the fly.
        """
        self.declare_parameters(
            namespace="",
            parameters=[(name, getattr(self, f"_{name}"), descriptor)
                for name, descriptor in self.CONFIGURATION_NAMES.items()]
        )
        self.add_on_set_parameters_callback(self.parameters_callback)

    def parameters_callback(self, params: List[Parameter]):
        """Callback for setting parameters

        :param params: _description_
        :type params: List[Parameter]
        :return: _description_
        :rtype: _type_
        """
        for param in params:
            if param.name in self.CONFIGURATION_NAMES:
                if param.type_ == Parameter.Type.DOUBLE:
                    setattr(self, f"_{param.name}", param.value)
                else:
                    self.get_logger().warn(f"Invalid type {param.type_}, ignoring request")

        dump = json.dumps(self.get_current_configuration(),
            sort_keys=True, indent=2)
        self.get_logger().info(f"Configuration after changes: {dump}")

        return SetParametersResult(successful=True)

    def car_cmd_callback(self, msg_car_cmd: Twist2DStamped):
        """A callback that reposponds to received `car_cmd` messages by
        calculating the corresponding wheel commands, taking into account the
        robot geometry, gain and trim factors, and the set limits.
        These wheel commands are then published for the motors to use.
        The resulting linear and angular velocities are also calculated
        and published.
        Args:
            msg_car_cmd (:obj:`Twist2DStamped`): desired car command
        """
        # INVERSE KINEMATICS PART

        # trim the desired commands such that they are within the limits:
        msg_car_cmd.v = self.trim(
            msg_car_cmd.v, low=-self._v_max,
            high=self._v_max)
        msg_car_cmd.omega = self.trim(
            msg_car_cmd.omega, low=-self._omega_max,
            high=self._omega_max
        )

        # assuming same motor constants k for both motors
        k_r = k_l = self._k

        # adjusting k by gain and trim
        k_r_inv = (self._gain + self._trim) / k_r
        k_l_inv = (self._gain - self._trim) / k_l

        omega_r = ((msg_car_cmd.v + 0.5 * msg_car_cmd.omega * self._baseline)
            / self._radius)
        omega_l = ((msg_car_cmd.v - 0.5 * msg_car_cmd.omega * self._baseline)
            / self._radius)

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = self.trim(u_r, -self._limit, self._limit)
        u_l_limited = self.trim(u_l, -self._limit, self._limit)

        # Put the wheel commands in a message and publish
        msg_wheels_cmd = WheelsCmdStamped()
        msg_wheels_cmd.header.stamp = msg_car_cmd.header.stamp
        msg_wheels_cmd.vel_right = u_r_limited
        msg_wheels_cmd.vel_left = u_l_limited
        self.pub_wheels_cmd.publish(msg_wheels_cmd)

        # FORWARD KINEMATICS PART

        # Conversion from motor duty to motor rotation rate
        omega_r = msg_wheels_cmd.vel_right / k_r_inv
        omega_l = msg_wheels_cmd.vel_left / k_l_inv

        # Compute linear and angular velocity of the platform
        v = (self._radius * omega_r + self._radius * omega_l) / 2.0
        omega = ((self._radius * omega_r - self._radius * omega_l)
            / self._baseline)

        # Put the v and omega into a velocity message and publish
        msg_velocity = Twist2DStamped()
        msg_velocity.header = msg_wheels_cmd.header
        msg_velocity.v = v
        msg_velocity.omega = omega
        self.pub_velocity.publish(msg_velocity)

    def save_calibration_callback(self, request, response):
        """Service request callback to save current configuration to file.
        """
        full_fp = calibration_utils.save_calibration(
            os.path.join(self.REL_CALIBRATION_DIR, f"{self._veh_name}.yaml"),
            self.get_current_configuration())

        msg = f"Calibration saved to {full_fp}"
        self.get_logger().info(msg)

        response.success = True
        response.message = msg
        return response

    @staticmethod
    def trim(value, low, high):
        """Trims a value to be between some bounds.
        Args:
            value: the value to be trimmed
            low: the minimum bound
            high: the maximum bound
        Returns:
            the trimmed value
        """
        return max(min(value, high), low)


def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode("kinematics_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
