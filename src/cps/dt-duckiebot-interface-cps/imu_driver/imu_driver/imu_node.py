#!/usr/bin/env python3

import os
import math
import rclpy

from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import Imu
from imu_driver.mpu9250 import mpu9250
from dt_calibration_utils import calibration_file


G = 9.80665
DEG2RAD = math.pi / 180


class IMUHandler(Node):
    CALIBRATION_REL_DIR = "imu/"

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._retrieve_parameters()
        self.current_state = True

        try:
            self.sensor = mpu9250(1)
            self.sensor.accel
            self.sensor.gyro
        except IOError:
            raise RuntimeError("IMU sensor not detected")
        
        self.pub = self.create_publisher(Imu, "imu", 10)
        self.timer = self.create_timer(1.0/self._polling_hz, self.publish_data)

    def _retrieve_parameters(self):
        self.declare_parameter("veh")
        self.declare_parameter("use_calibration")
        self.declare_parameter("polling_hz")
        self.declare_parameter("ang_vel_offset")
        self.declare_parameter("accel_offset")

        self._veh = self.get_parameter("veh").get_parameter_value().string_value
        self._use_calibration = self.get_parameter("use_calibration")\
            .get_parameter_value().bool_value
        self._polling_hz = self.get_parameter("polling_hz")\
            .get_parameter_value().integer_value

        if self._use_calibration:
            data, fp = calibration_file.read_calibration(
                os.path.join(IMUHandler.CALIBRATION_REL_DIR, f"{self._veh}.yaml"))
            if data is None:
                self.get_logger().warn(f"Unable to read calibration file {fp}."\
                    " Using default parameters.")
            else:
                params_override = [
                    Parameter(key, Parameter.Type.INTEGER_ARRAY, data[key]) 
                    for key in ("ang_vel_offset", "accel_offset")]
                self.set_parameters(params_override)
                calibration_time = data.get("calibration_time")
                self.get_logger().info("Overriding parameters with calibration"\
                    f" file: {fp}, timestamp of file: {calibration_time}")

        self._ang_vel_offset = self.get_parameter("ang_vel_offset")\
            .get_parameter_value().integer_array_value
        self._accel_offset = self.get_parameter("accel_offset")\
            .get_parameter_value().integer_array_value

    def publish_data(self):
        try:
            a = self.sensor.accel
            g = self.sensor.gyro
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.angular_velocity.x = g[0] * DEG2RAD - self._ang_vel_offset[0]
            msg.angular_velocity.y = g[1] * DEG2RAD - self._ang_vel_offset[1]
            msg.angular_velocity.z = g[2] * DEG2RAD - self._ang_vel_offset[2]
            msg.linear_acceleration.x = a[0] * G - self._accel_offset[0]
            msg.linear_acceleration.y = a[1] * G - self._accel_offset[1]
            msg.linear_acceleration.z = a[2] * G - self._accel_offset[2]
            for i in range(0, 9):
                msg.angular_velocity_covariance[i] = 0
                msg.linear_acceleration_covariance[i] = 0
                msg.orientation_covariance[i] = -1

            self.pub.publish(msg)
        except IOError as error:
            self.get_logger().error(f"Unable to read Imu sensor due to"\
                f" I/O error: {error}")


def main(args=None):
    rclpy.init(args=args)
    imu_node = IMUHandler(node_name="imu_node")
    try:
        rclpy.spin(imu_node)
    except KeyboardInterrupt:
        pass
    finally:
        imu_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

