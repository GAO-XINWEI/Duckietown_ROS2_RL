#!/usr/bin/env python3

import rclpy

from rclpy.node import Node
from battery_drivers.battery import Battery
from dt_health_interfaces_cps.srv import BatteryInfo
from dt_health_interfaces_cps.msg import BatteryHealth


class BatteryHealthNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._get_parameters()

        self._battery = Battery(callback=None, logger=self.get_logger())
        self._battery.start()

        # Publisher for battery health data
        self._health_pub = self.create_publisher(BatteryHealth,
            "~/battery_health", 1)

        # Service to get information about the battery
        self._info_srv = self.create_service(BatteryInfo,
            "~/battery_info", self._battery_info_cb)

        self._timer = self.create_timer(self._publish_period, self._timer_cb)

    def _get_parameters(self):
        self.declare_parameter("publish_period", 5.0)

        self._publish_period = self.get_parameter("publish_period")\
            .get_parameter_value().double_value

    def _timer_cb(self):
        data = self._battery.data

        if data is None:
            msg = BatteryHealth(time=self.get_clock().now().to_msg(),
                present=False)
        else:
            msg = BatteryHealth(**data)
            msg.time = self.get_clock().now().to_msg()
        self._health_pub.publish(msg)

    def _battery_info_cb(self, request, response):
        data = self._battery.info
        if data is None:
            response.valid = False
        else:
            response.valid = True
            response.version = data['version']
            response.boot_version = data['boot']['version']
            response.boot_pcb_version = data['boot']['pcb_version']
            response.boot_date = data['boot']['date']
            response.serial_number = data['serial_number'] 
        return response


def main(args=None):
    rclpy.init(args=args)
    battery_health_node = BatteryHealthNode(node_name="battery_health_node")
    try:
        rclpy.spin(battery_health_node)
    except KeyboardInterrupt:
        pass
    finally:
        battery_health_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
