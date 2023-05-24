#!/usr/bin/env python3

import yaml
import rclpy

from rclpy.node import Node

from dt_interfaces_cps.srv import ChangePattern


class LEDEmitterTestNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.get_launch_params()
        self.get_patterns()

        if self.signal not in self.signals:
            raise ValueError("Signal to set not valid.")

        self.cli = self.create_client(ChangePattern, "~/set_pattern")

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("service not available, waiting again...")

        response = self.send_request(self.signal)
        self.get_logger().info(f'Sent request: {self.signal}, response: {response}')

    def get_launch_params(self):
        self.declare_parameter("config_file_path")
        self.declare_parameter("signal")

        self.protocol_filename = self.get_parameter("config_file_path").get_parameter_value()\
            .string_value
        self.signal = self.get_parameter("signal").get_parameter_value()\
            .string_value

    def get_patterns(self):
        with open(self.protocol_filename, 'r') as f:
            config = yaml.safe_load(f)
        self.signals = [signal for signal in config['LED_protocol']['signals']]

    def send_request(self, signal: str):
        req = ChangePattern.Request()
        req.pattern_name.data = signal
        self.future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    node = LEDEmitterTestNode(node_name="led_emitter_test_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
