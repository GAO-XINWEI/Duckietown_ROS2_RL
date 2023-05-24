#!usr/bin/env python3

import os
import math
import rclpy

from dataclasses import dataclass

try:
    import smbus2 as smbus
except ImportError:
    try:
        import smbus
    except ImportError:
        raise RuntimeError("At least one library between `smbus` and `smbus2`"\
            " must be installed.")

from sensor_msgs.msg import Range
from std_msgs.msg import Header
from rclpy.node import Node

from vl53l0x_driver.VL53L0X import VL53L0X, Vl53l0xAccuracyMode


class SensorNotFoundError(Exception):
    pass


@dataclass
class ToFAccuracy:
    mode: Vl53l0xAccuracyMode
    timing_budget: float
    max_range: float
    # the following are taken from the sensor's datasheet
    min_range: float = 0.05
    fov: float = math.radians(25)

    @staticmethod
    def from_string(mode: str):
        ms = 1/1000
        return {
            "GOOD": ToFAccuracy(Vl53l0xAccuracyMode.GOOD, 33*ms, 1.2),
            "BETTER": ToFAccuracy(Vl53l0xAccuracyMode.BETTER, 66*ms, 1.2),
            "BEST": ToFAccuracy(Vl53l0xAccuracyMode.BEST, 200*ms, 1.2),
            "LONG_RANGE": ToFAccuracy(Vl53l0xAccuracyMode.LONG_RANGE, 33*ms, 2.0),
            "HIGH_SPEED": ToFAccuracy(Vl53l0xAccuracyMode.HIGH_SPEED, 20*ms, 1.2)
        }[mode]


class ToFNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._retrieve_parameters()
        self._accuracy = ToFAccuracy.from_string(self._mode)

        self.smbus = smbus.SMBus(self._i2c_bus)
        self._detect_i2c_tof(i2c_addr=self._i2c_address)
        # Create a VL53L0X sensor handler
        self._sensor = VL53L0X(
            i2c_bus=self._i2c_bus, i2c_address=self._i2c_address)
        self._sensor.open()

        # Create publishers
        self._pub = self.create_publisher(
            Range,
            "~/range",
            1)
        
        # start ranging
        self._sensor.start_ranging(self._accuracy.mode)
        max_frequency = min(self._frequency,
            int(1.0/self._accuracy.timing_budget))
        if self._frequency > max_frequency:
            self.get_logger().warn(f"Frequency of {self._frequency}Hz not"\
                " supported. The selected mode {self._mode} has timing budget"\
                " of {self._accuracy.timing_budget}s, which yields maximum"\
                " frequency of {max_frequency}Hz.")
            self._frequency = max_frequency
        self.get_logger().info(f"Frequency set to {self._frequency}Hz.")

        # create timers
        self.timer = self.create_timer(1.0/max_frequency, self._timer_cb)

    def _retrieve_parameters(self):
        # get parameters
        self.declare_parameter("veh")
        self.declare_parameter("bus")
        self.declare_parameter("address", 0x29)
        self.declare_parameter("sensor_name")
        self.declare_parameter("frequency", 10)
        self.declare_parameter("mode", value="BETTER")
        self.declare_parameter("mux_addr", 0x70)
        self.declare_parameter("mux_channel")

        self._veh = self.get_parameter("veh").get_parameter_value().string_value
        self._i2c_bus = self.get_parameter("bus").get_parameter_value()\
            .integer_value
        self._i2c_address = self.get_parameter("address").get_parameter_value()\
            .integer_value
        self._sensor_name = self.get_parameter("sensor_name")\
            .get_parameter_value().string_value
        self._frequency = int(max(1, self.get_parameter("frequency")\
            .get_parameter_value().integer_value))
        self._mode = self.get_parameter("mode")\
            .get_parameter_value().string_value
        self._mux_addr = self.get_parameter("mux_addr")\
            .get_parameter_value().integer_value
        self._mux_channel = self.get_parameter("mux_channel")\
            .get_parameter_value().integer_value

    def _detect_i2c_tof(self, i2c_addr: int):
        try:
            self.select_tof()
            self.smbus.read_byte(i2c_addr)
        except OSError:
            raise SensorNotFoundError(f"Cannot find ToF sensor at address {i2c_addr}")

    def _timer_cb(self):
        # detect range
        self.select_tof()
        distance_mm = self._sensor.get_distance()
        # pack observation into a message
        msg = Range(header=Header(stamp=self.get_clock().now().to_msg(),
            frame_id=f"{self._veh}/tof/{self._sensor_name}"),
            radiation_type=Range.INFRARED,
            field_of_view=self._accuracy.fov,
            min_range=self._accuracy.min_range,
            max_range=self._accuracy.max_range,
            range=distance_mm/1000)
        # publish
        self._pub.publish(msg)

    def select_tof(self):
        """
        Selects which port of the multiplexer is currently being used.
        :return: None
        :raises: IOError if it is unable to communicate with the front bumper multiplexer
        """
        self.smbus.write_byte(self._mux_addr, 1 << self._mux_channel)
    
    def on_shutdown(self):
        try:
            self.select_tof()
            self._sensor.stop_ranging()
        except BaseException:
            pass


def main(args=None):
    rclpy.init(args=args)
    tof_node = ToFNode(node_name="tof_node")
    try:
        rclpy.spin(tof_node)
    except SensorNotFoundError as error:
        tof_node.get_logger().fatal(f"Check ToF sensor, {error}")
    except KeyboardInterrupt:
        pass
    finally:
        tof_node.on_shutdown()
        tof_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
