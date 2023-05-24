#!/usb/bin/env python3

import rclpy

from rclpy.node import Node
from threading import Thread, Semaphore

from hardware_stats.stats_parser import *
from dt_health_interfaces_cps.msg import HardwareStats as HardwareStatsMsg
from dt_device_utils import (
    DeviceHardwareBrand,
    get_device_hardware_brand
)


class HardwareNotSupportedException(Exception):
    pass


class HardwareStatsNode(Node):

    _SUPPORTED_HARDWARE = [DeviceHardwareBrand.JETSON_NANO]

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._hardware_type = get_device_hardware_brand()
        if self._hardware_type not in HardwareStatsNode._SUPPORTED_HARDWARE:
            raise HardwareNotSupportedException("Hardware currently unsupported")

        self._get_parameters()

        # Data to be published
        self._msg = HardwareStatsMsg(
            hardware_type=HardwareStatsMsg.HARDWARE_JETSON)

        self._semaphore = Semaphore(value=1)

        self._worker = Thread(target=self._read_hardware)
        self.start()

        # Create publisher
        self._pub = self.create_publisher(
            HardwareStatsMsg, "~/hardware_stats", 1)
        # Timer to publish on topic
        self._timer = self.create_timer(self._pub_period, self._timer_cb)

    def _get_parameters(self):
        self.declare_parameter("publish_period", 1.0)

        self._pub_period = self.get_parameter("publish_period")\
            .get_parameter_value().double_value

    def _timer_cb(self):
        self._semaphore.acquire()
        self._pub.publish(self._msg)
        self._semaphore.release()

    def start(self):
        self._worker.start()

    def _read_hardware(self):
        self.get_logger().info("Starting to read hardware")
        for line in TegraStatsParser().execute():
            data = TegraStatsParser.parse_data(line)
            tmp = HardwareStatsMsg(timestamp=self.get_clock().now().to_msg(),
                hardware_type=self._hardware_type,
                ram_used=float(data.ram.used),
                ram_total=float(data.ram.total),
                free_ram_blocks=int(data.ram.free_ram_blocks),
                iram_used=float(data.iram.used),
                iram_total=float(data.iram.total),
                iram_size=float(data.iram.size),
                swap_used=float(data.swap.used),
                swap_total=float(data.swap.total),
                swap_cached=float(data.swap.cached),
                cpus_num=int(data.cpus.num),
                cpu_names=list(data.cpus.frequencies.keys()) if data.cpus.frequencies is not None else list(),
                cpu_frequencies=list(data.cpus.frequencies.values()) if data.cpus.frequencies is not None else list(),
                cpu_loads=list(data.cpus.load.values()) if data.cpus.load is not None else list(),
                gr3d_used=float(data.gr3d.used),
                gr3d_frequency=float(data.gr3d.frequency),
                emc_used=float(data.emc.used),
                emc_frequency=float(data.emc.frequency),
                temp_nums=int(data.temps.nums),
                temp_names=list(data.temps.temps.keys()) if data.temps.temps is not None else list(),
                temp_values=list(data.temps.temps.values()) if data.temps.temps is not None else list(),
                pwr_nums=int(data.pwr.nums),
                pwr_names=list(data.pwr.avg_pwr.keys()) if data.pwr.avg_pwr is not None else list(),
                pwr_inst=list(data.pwr.inst_pwr.values()) if data.pwr.inst_pwr is not None else list(),
                pwr_avg=list(data.pwr.avg_pwr.values()) if data.pwr.avg_pwr is not None else list())
            self._semaphore.acquire()
            self._msg = tmp
            self._semaphore.release()


def main(args=None):
    rclpy.init(args=args)
    node = HardwareStatsNode(node_name="hardware_stats_node")
    try:
        rclpy.spin(node)
    except HardwareNotSupportedException as error:
        node.get_logger().fatal(f"{error}")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
