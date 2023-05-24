#!/usr/bin/env python3

import rclpy
import time

from std_msgs.msg import String
from rclpy.node import Node

from dt_interfaces_cps.msg import ButtonEvent as ButtonEventMsg
from button_driver.button import ButtonEvent, ButtonDriver
from dt_device_utils.device import shutdown_device
from dt_interfaces_cps.srv import ChangePattern


class ButtonDriverNode(Node):
    _TIME_DOUBLE_CLICK_S = 0.1
    _TIME_HOLD_3S = 3
    _TIME_HOLD_10S = 10

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._get_parameters()

        # create publisher
        self._pub = self.create_publisher(ButtonEventMsg, "~/event", 1)

        # create service client to front and back LEDs
        self._srv = self.create_client(
            ChangePattern, "led_emitter_node/set_pattern")
        srv_try_count = 0
        while not self._srv.wait_for_service(timeout_sec=1.0):
            if srv_try_count > 3:
                self.get_logger().warn("No `led_emitter_node/set_pattern`"\
                " service found - retry count exceeded")
                self._srv.destroy()
                self._srv = None
                break
            self.get_logger().warn("`led_emitter_node/set_pattern` service not"\
                " not available, waiting again...")
            srv_try_count += 1

        # create button driver
        self._button = ButtonDriver(self._led_gpio_pin, self._signal_gpio_pin,
            self._event_cb)
        self._button.led.on()

        # create event holder
        self._ongoing_event = None

    def _get_parameters(self):
        self.declare_parameter("led_gpio_pin")
        self.declare_parameter("signal_gpio_pin")

        self._led_gpio_pin = self.get_parameter("led_gpio_pin")\
            .get_parameter_value().integer_value
        self._signal_gpio_pin = self.get_parameter("signal_gpio_pin")\
            .get_parameter_value().integer_value

    def _event_cb(self, event: ButtonEvent):
        # create partial event
        if event == ButtonEvent.PRESS:
            # create new partial event
            self._ongoing_event = time.time()
            return
        # this is a RELEASE event
        if self._ongoing_event is None:
            # we missed it, well, next time!
            return
        # create new full event
        # todo: Maybe its a better ideal to use the ROS time?
        duration = time.time() - self._ongoing_event
        # clear ongoing event
        self._ongoing_event = None
        # analyze event
        # - single click
        if duration < 0.5:
            self._publish(ButtonEventMsg.EVENT_SINGLE_CLICK)
            self._react(ButtonEventMsg.EVENT_SINGLE_CLICK)
            self.get_logger().info("Button single click")
            return
        # - held for 3 secs
        if (ButtonDriverNode._TIME_HOLD_3S < duration
            < 2 * ButtonDriverNode._TIME_HOLD_3S):
            # publish a display showing shutdown confirmation
            self._publish(ButtonEventMsg.EVENT_HELD_3SEC)
            self._react(ButtonEventMsg.EVENT_HELD_3SEC)
            self.get_logger().info("Button held for 3 seconds.")
            return
        if ButtonDriverNode._TIME_HOLD_10S < duration:
            self._publish(ButtonEventMsg.EVENT_HELD_10SEC)
            self._react(ButtonEventMsg.EVENT_HELD_10SEC)
            self.get_logger().info("Button held for 10 seconds.")
            return

    def _publish(self, event: int):
        msg = ButtonEventMsg()
        msg.event = event
        self._pub.publish(msg)

    def _react(self, event: int):
        if event == ButtonEventMsg.EVENT_HELD_10SEC:
            # blink top power button as a confirmation, too
            self._button.led.confirm_shutdown()
            
            # turn off front and back LEDs
            if self._srv is not None:
                req = ChangePattern.Request()
                req.pattern_name = String(data="LIGHT_OFF")
                future = self._srv.call_async(req)
                rclpy.spin_until_future_complete(self, future)

            res = shutdown_device()
            if not res:
                self.get_logger().error(
                    "Could not initialize the shutdown sequence")

    def on_shutdown(self):
        if hasattr(self, '_button'):
            self._button.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ButtonDriverNode(node_name="button_driver_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
