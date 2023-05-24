#!/usr/bin/env python3

import time
import rclpy
import Adafruit_SSD1306 

from PIL import Image
from typing import List
from rclpy.node import Node
from threading import Thread
from dataclasses import dataclass
from PIL import (
    Image,
    ImageDraw,
    ImageFont
)

from dt_interfaces_cps.msg import ButtonEvent as ButtonEventMsg


class FontNotFoundError(Exception):
    pass


@dataclass
class Font:
    name: str
    path: str
    size: int


@dataclass
class CanvasConstants:
    padding: int
    top: int
    bottom: int
    height: int
    width: int


# List of predefined fonts with suitable font sizes to be used on the display
PREDEFINED_FONTS = [
    Font(name="Ubuntu-M",
        path="/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", size=8)
]


class DisplayNode(Node):

    NUMBER_OF_PAGES = 2

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._retrieve_parameters()
        self._font = self.load_font(PREDEFINED_FONTS)
        self._shutting_down = False
        self._current_page = 0
        self._button_sub = self.create_subscription(
            ButtonEventMsg,
            "button_driver_node/event",
            self.button_callback,
            1)

        self._disp = Adafruit_SSD1306.SSD1306_128_32(
            rst=None, i2c_bus=self._i2c_bus, gpio=1)
        self._canvas_constants = DisplayNode.define_canvas_constants(self._disp)
        self._image = Image.new('1',
            (self._canvas_constants.width, self._canvas_constants.height))
        self._draw = ImageDraw.Draw(self._image)
        self._draw.rectangle(
            (0, 0, self._canvas_constants.width, self._canvas_constants.height),
            outline=0, fill=0)
        self._disp.begin()
        self.clear_display()

        # Start a new thread to draw on the screen
        self._worker = Thread(target=self.start_drawing)
        self._worker.start()


    def _retrieve_parameters(self):
        self.declare_parameter("veh")
        self.declare_parameter("bus", 1)
        self.declare_parameter("address", 0x3C)
        self.declare_parameter("refresh_frequency", 1.0)

        self._veh = self.get_parameter("veh").get_parameter_value().string_value
        self._i2c_bus = self.get_parameter("bus")\
            .get_parameter_value().integer_value
        self._i2c_address = self.get_parameter("address")\
            .get_parameter_value().integer_value
        self._refresh_frequency = self.get_parameter("refresh_frequency")\
            .get_parameter_value().double_value
        
        self.get_logger().info(f"""Parameters: 
        veh: {self._veh},
        bus: {self._i2c_bus},
        address: {self._i2c_address},
        refresh frequency: {self._refresh_frequency}""")

    def load_font(self, fonts: List[Font]):
        """Try to load fonts from file system.
        Will raise `FontNotFoundError` exception if no fonts could be loaded.
        """
        for font in fonts:
            try:
                return ImageFont.truetype(font.path, font.size)
            except IOError as _:
                pass
        raise FontNotFoundError("No suitable fonts could be found.")

    @staticmethod
    def define_canvas_constants(display) -> CanvasConstants:
        height = display.height
        width = display.width
        padding = -2
        bottom = height - padding
        return CanvasConstants(padding=padding, top=padding,
            bottom=bottom, height=height, width=width)

    def start_drawing(self):
        width = self._canvas_constants.width
        height = self._canvas_constants.height
        top = self._canvas_constants.top
        font = self._font
        x = 0
        while True and not self._shutting_down:
            self._draw.rectangle((0, 0, width, height), outline=0, fill=0)
            
            if self._current_page == 0:
                self._draw.text((x, top), "Page 1", font=font, fill=255)
            elif self._current_page == 1:
                self._draw.text((x, top+25), "Page 2", font=font, fill=255)

            self._disp.image(self._image)
            self._disp.display()
            time.sleep(1.0/self._refresh_frequency)

    def button_callback(self, msg: ButtonEventMsg):
        if msg.event == ButtonEventMsg.EVENT_SINGLE_CLICK:
            self._current_page = (self._current_page + 1) % DisplayNode.NUMBER_OF_PAGES

    def clear_display(self):
        self._disp.clear()
        self._disp.display()

    def on_shutdown(self):
        self._shutting_down = True
        self.clear_display()


def main(args=None):
    rclpy.init(args=args)
    node = DisplayNode(node_name="display_node")
    try:
        rclpy.spin(node)
    except FontNotFoundError as error:
        node.get_logger().fatal(f"{error}")
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
