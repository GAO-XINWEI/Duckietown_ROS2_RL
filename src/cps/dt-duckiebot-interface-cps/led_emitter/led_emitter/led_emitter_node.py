#!/usr/bin/env python3
import rclpy
import yaml

from rclpy.node import Node
from std_msgs.msg import String

from dt_interfaces_cps.srv import (
    SetCustomLEDPattern,
    ChangePattern,
)

from led_emitter.rgb_led import RGB_LED


class LEDEmitterNode(Node):
    """Node for controlling LEDs.
    Calls the low-level functions of class :obj:`RGB_LED` that creates the PWM
    signal used to change the color of the LEDs. The desired behavior is
    specified by the LED index (Duckiebots and watchtowers have multiple
    of these) and a pattern.
    A pattern is a combination of colors and blinking frequency.
    Duckiebots have 5 LEDs that are indexed and positioned as following:
    +------------------+------------------------------------------+
    | Index            | Position (rel. to direction of movement) |
    +==================+==========================================+
    | 0                | Front left                               |
    +------------------+------------------------------------------+
    | 1                | Rear left                                |
    +------------------+------------------------------------------+
    | 2                | Top / Front middle                       |
    +------------------+------------------------------------------+
    | 3                | Rear right                               |
    +------------------+------------------------------------------+
    | 4                | Front right                              |
    +------------------+------------------------------------------+
    A pattern is specified via 5 parameters:
    - its name
    - frequency: blinking frequency in Hz, should be set to 0 for a solid
      (non-blinking) behavior
    - color_list: a list of 5 colour names (see below), one for each LED ordered
      as above, or a single string with a single color name that would be
      applied to all LEDs
    - frequency_mask: a list of 5 binary flags (0 or 1) that specify which of
      the LEDs should be blinking, used only if the frequency is not 0.
      The LEDs with the flag set to 0, will maintain their solid color.
      The defaut patterns are defined in the `LED_protocol.yaml` configuration
      file for the node. Currently supported colors are: `green`, `red`, `blue`,
      `white`, `yellow`, `purple`, `cyan`, `pink`, `switchedoff`.
      More colors can be defined in the node's configuration file.
    Examples:
        To change the pattern to one of the predefined patterns (you can see
        them using `rosparam list`) use a variant of the following::
            rosservice call /HOSTNAME/led_emitter_node/set_pattern
            "pattern_name: {data: RED}"
        Other pre-defined patterns you can use are:
            `WHITE`, `GREEN`, `BLUE`,
            `LIGHT_OFF`, `CAR_SIGNAL_PRIORITY`,
            `CAR_SIGNAL_SACRIFICE_FOR_PRIORITY`,
            `CAR_SIGNAL_SACRIFICE_FOR_PRIORITY`,
            `CAR_SIGNAL_SACRIFICE_FOR_PRIORITY`, `CAR_DRIVING`.
        To add a custom pattern and switch to it use a variant of the
            following::rosservice call
                /HOSTNAME/led_emitter_node/set_custom_pattern
            "pattern: {color_list: ['green','yellow','pink','orange','blue'],
            color_mask: [1,1,1,1,1], frequency: 1.0,
            frequency_mask: [1,0,1,0,1]}"
    Configuration:
        ~LED_protocol (nested dictionary): Nested dictionary that describes
            the LED protocols (patterns). The default can be seen in the
            `LED_protocol.yaml` configuration file for the node.
        ~LED_scale (:obj:`float`): A scaling factor (between 0 and 1) that is
            applied to the colors in order to reduce the overall LED brightness,
            default is 0.8.
        ~channel_order (:obj:`str`): A string that controls the order in which
            the 3 color channels should be communicated to the LEDs. Should be
            one of 'RGB', `RBG`, `GBR`, `GRB`, `BGR`, `BRG`. Typically
            for a duckiebot this should be the default `RGB` and for traffic
            lights should be `GRB`, default is `RGB`.
    Publishers:
        ~current_led_state (:obj:`String` message): Publishes the name of the
            current pattern used. Published
            only when the selected pattern changes.
    Services:
        ~set_custom_pattern: Allows setting a custom protocol. Will be named
            `custom`. See an example of a call in :obj:`srvSetCustomLEDPattern`.
            input:
                pattern (:obj:`LEDPattern` message): The desired new LEDPattern
        ~set_pattern: Switch to a different pattern protocol.
            input:
                pattern_name (:obj:`String` message): The new pattern name,
                    should match one of the patterns in
                    the `LED_protocol` parameter (or be `custom` if a custom
                    pattern has been defined via a call to 
                    the `~change_led` service.
    """

    CONFIG_FILE_PATH_PARAM_NAME = "config_file_path"
    ROBOT_TYPE_PARAM_NAME = "robot_type"

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.led = RGB_LED()

        self.declare_parameter(LEDEmitterNode.CONFIG_FILE_PATH_PARAM_NAME)
        self.declare_parameter(LEDEmitterNode.ROBOT_TYPE_PARAM_NAME)

        self._load_config(self.get_parameter(
            LEDEmitterNode.CONFIG_FILE_PATH_PARAM_NAME
                ).get_parameter_value().string_value)
        self.robot_type = self.get_parameter(
            LEDEmitterNode.ROBOT_TYPE_PARAM_NAME).get_parameter_value().string_value
        self.get_logger().info(f"Robot type: {self.robot_type}")

        # Initialize LEDs to be off
        self.switch = True
        self.pattern = [[0, 0, 0]] * 5
        self.frequency_mask = [0] * 5
        self.current_pattern_name = 'LIGHT_OFF'
        self.changePattern(self.current_pattern_name)

        # Initialize the timer
        self.frequency = 1.0/self._LED_protocol["signals"]["CAR_SIGNAL_A"]["frequency"]
        self.is_on = False
        self.cycle_timer = self.create_timer(self.frequency/2.0,
            self._cycle_timer)

        # Publishers
        self.pub_state = self.create_publisher(
            String, "~/current_led_state", 1)

        # Services
        self.srv_set_LED = self.create_service(
            SetCustomLEDPattern,
            "~/set_custom_pattern", 
            self.srvSetCustomLEDPattern)

        self.srv_set_pattern = self.create_service(
            ChangePattern,
            "~/set_pattern",
            self.srvSetPattern)

        # Scale intensity of the LEDs
        for name, c in self._LED_protocol["colors"].items():
            for i in range(3):
                c[i] = c[i] * self._LED_scale

        # Remap colors if robot does not have an RGB ordering
        if self._channel_order[self.robot_type] != "RGB":
            protocol = self._LED_protocol
            for name, col in self._LED_protocol["colors"].items():
                protocol["colors"][name] = self.remapColors(col)

            # Update LED_protocol
            self._LED_protocol = protocol
            self.get_logger().warn(f"""
                Colors remapped to {str(self._channel_order[self.robot_type])}
            """)

        # Turn on the LEDs
        self.changePattern("WHITE")

        self.get_logger().info("Initialized.")


    def _load_config(self, filepath: str):
        """Load the YAML configuration file"""
        with open(filepath, 'r') as stream:
            config =  yaml.safe_load(stream)
        self._LED_protocol = config["LED_protocol"]
        self._LED_scale = config["LED_scale"]
        self._channel_order = config["channel_order"]

    def srvSetCustomLEDPattern(self, request, response):
        """Service to set a custom pattern.
            Sets the LEDs to a custom pattern. The :obj:`LEDPattern`
            message from :obj:`duckietown_msgs` is used for that.
            Args:
                req (SetCustomLEDPatternRequest): the requested pattern
        """
        # Update the protocol
        protocol = self._LED_protocol
        protocol['signals']['custom'] = {
            'color_mask': request.pattern.color_mask,
            'color_list': request.pattern.color_list,
            'frequency_mask': request.pattern.frequency_mask,
            'frequency': request.pattern.frequency
        }
        # update the LED_protocol
        self._LED_protocol = protocol

        self.get_logger().info("Custom pattern updated")
        self.get_logger().debug(f"""
            color_mask: {self._LED_protocol['signals']['custom']['color_mask']}
            color_list: {self._LED_protocol['signals']['custom']['color_list']}
            frequency_mask: {self._LED_protocol['signals']['custom']['frequency_mask']}
            frequency: {self._LED_protocol['signals']['custom']['frequency']}
        """)

        # Perform the actual change
        self.changePattern("custom")

        return response

    def _cycle_timer(self):
        """Timer.
            Calls updateLEDs according to the frequency of the current pattern.
            Args:
                event (TimerEvent): event generated by the timer.
        """
        self.updateLEDs()

    def updateLEDs(self):
        """Switches the LEDs to the requested signal.
            If the pattern is static, changes the color of LEDs according to
            the color specified in self.color_list. If a nonzero frequency is
            set, toggles on/off the LEDs specified on self.frequency_mask.
        """
        # Do nothing if inactive
        if not self.switch:
            return
        elif not self.frequency:
            # No oscillation
            for i in range(5):
                colors = self.pattern[i]
                self.led.setRGB(i, colors)
        else:
            # Oscillate
            if self.is_on:
                for i in range(5):
                    if self.frequency_mask[i]:
                        self.led.setRGB(i, [0,0,0])
                self.is_on = False
            else:
                for i in range(5):
                    colors = self.pattern[i]
                    self.led.setRGB(i, colors)
                self.is_on = True

    def srvSetPattern(self, request, response):
        """Changes the current pattern according to the pattern name
            sent in the message.
            Args:
                msg (String): requested pattern name
        """
        self.changePattern(request.pattern_name.data)
        return response

    def changePattern(self, pattern_name: str):
        if not pattern_name:
            return
        if (self.current_pattern_name == pattern_name 
            and pattern_name != "custom"):
            return

        if pattern_name.strip("'").strip('"') in self._LED_protocol["signals"]:
            self.current_pattern_name = pattern_name
        else:
            self.get_logger().warn(f"""Pattern name {pattern_name} not found in
                list of patterns. Change of pattern not executed""")
            return
        
        # Extract the color from the protocol config file
        color_list = self._LED_protocol["signals"][pattern_name]["color_list"]

        if type(color_list) is str:
            self.pattern = [self._LED_protocol["colors"][color_list]]*5
        else:
            if len(color_list) != 5:
                self.get_logger().warn("""The color list should be a string or a
                    list of length 5. Change of pattern not executed""")
                return

            self.pattern = [[0,0,0]] * 5
            for i in range(len(color_list)):
                if isinstance(color_list[i], str):
                    self.pattern[i] = self._LED_protocol["colors"][color_list[i]]
                elif (isinstance(color_list[i], list)
                    and len(color_list[i]) == 3):
                    self.pattern[i] = color_list[i]
                    self.pattern[i] = [
                        max(0, min(c, 255)) for c in self.pattern[i]]
                else:
                    self.get_logger().error(f"""
                        LEDs color passed as RGB values must be expressed as
                        lists of 3 values from range [0, 255].""")
                    return
        
        # Extract the frequency from the protocol
        self.frequency_mask = self._LED_protocol["signals"][pattern_name]["frequency_mask"]
        self.frequency = self._LED_protocol["signals"][pattern_name]["frequency"]

        # If static behavior, updated LEDs
        if self.frequency == 0:
            self.updateLEDs()
        
        # Anyway modify the frequency (to stop timer if static)
        self.changeFrequency()

        # Publish current pattern
        current_pattern_msg = String()
        current_pattern_msg.data = self.current_pattern_name
        self.pub_state.publish(current_pattern_msg)

    def changeFrequency(self):
        """Changes current frequency of LEDs
            Stops the current cycle_timer, and starts a new one with the
            frequency specified in self.frequency. If the frequency is zero,
            stops the callback timer.
        """
        if self.frequency == 0:
            self.cycle_timer.destroy()
        else:
            try:
                self.cycle_timer.destroy()
                d = 1.0/(2.0*self.frequency)
                self.cycle_timer = self.create_timer(d, self._cycle_timer)
            except ValueError as error:
                self.frequency = None
                self.current_pattern_name = None

    def remapColors(self, color):
        """
        Remaps a color from RGB to the channel ordering currently set in the
        `channel_order` configuration parameter.
        Args:
            color (:obj:`list` of :obj:`float`): A color triplet
        Returns:
            :obj:`list` of :obj:`float`: The triplet with reordered channels
        """
        # Verify that the requested reordering is valid
        allowed_orderings = ['RGB', 'RBG', 'GBR', 'GRB', 'BGR', 'BRG']
        requested_ordering = self._channel_order[self.robot_type]
        if requested_ordering not in allowed_orderings:
            self.get_logger().warn(
                f"""The current channel order {requested_ordering}
                is not supported, use one of {str(allowed_orderings)}.
                The remapping was not performed.""")
            return color

        reordered_triplet = list()
        rgb_map = {'R':0, 'G':1, 'B':2}
        for channel_color in requested_ordering:
            reordered_triplet.append(color[rgb_map[channel_color]])

        return reordered_triplet

    def on_shutdown(self):
        """Shutdown procedure.
        At shutdown, changes the LED pattern to `LIGHT_OFF`.
        """
        # Turn off the lights when the node dies
        self.get_logger().info("Shutting down. Turning LEDs off.")
        self.changePattern("LIGHT_OFF")


def main(args=None):
    rclpy.init(args=args)

    led_emitter_node = LEDEmitterNode(node_name="led_emitter")
    rclpy.get_default_context().on_shutdown(led_emitter_node.on_shutdown)
    rclpy.spin(led_emitter_node)

    led_emitter_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
