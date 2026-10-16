"""Evaluate actual publication and driver math without importing hardware drivers."""
import ast
import importlib.util
import math
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('discrete_pwm', ROOT / 'rl_control/discrete_pwm.py')
presets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(presets)
DRIVER = ROOT.parents[1] / 'dt-duckiebot-interface-cps/wheels_driver/wheels_driver/dagu_wheels_driver.py'


def method(path, name):
    return next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def load_function(node, namespace):
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<test>', 'exec'), namespace)
    return namespace[node.name]


class PWMTests(unittest.TestCase):
    def test_roundtrip_through_actual_driver_and_zero_stop(self):
        tree = ast.parse(DRIVER.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        constants = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
                     if isinstance(n, ast.Assign)}
        driver = types.SimpleNamespace(**constants)
        convert = load_function(method(DRIVER, '_pwm_value'), dict(fabs=math.fabs, floor=math.floor))
        self.assertEqual((driver.LEFT_MOTOR_MIN_PWM, driver.RIGHT_MOTOR_MIN_PWM), (120, 120))
        self.assertEqual((driver.LEFT_MOTOR_MAX_PWM, driver.RIGHT_MOTOR_MAX_PWM), (255, 255))
        for action_id, expected in presets.TARGET_PWM.items():
            left, right = presets.discrete_wheel_commands(action_id)
            self.assertTrue(0 < left <= 1 and 0 < right <= 1)
            self.assertEqual((convert(driver, left, 120, 255), convert(driver, right, 120, 255)), expected)
        self.assertEqual(convert(driver, 0.0, 120, 255), 0)
        self.assertEqual(presets.pwm_to_command(0), 0)

    def test_invalid_action_and_pwm_rejected(self):
        for action_id in (-1, 3, None):
            with self.assertRaises(ValueError):
                presets.discrete_wheel_commands(action_id)
        for pwm in (1, 119, 256):
            with self.assertRaises(ValueError):
                presets.pwm_to_command(pwm)

    def test_actual_discrete_publication_without_dataset(self):
        class Message:
            def __init__(self):
                self.header = types.SimpleNamespace(stamp=None)
        stamp = types.SimpleNamespace(sec=1, nanosec=2)
        clock = types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: stamp))
        namespace = dict(ActionFloat32=Message, ActionFloat32Array2=Message,
                         WheelsCmdStamped=Message, GAIN=0.5, TRIM=0.0, K=27.0,
                         WHEEL_DIST=0.102, RADIUS=0.0318, LIMIT=1.0,
                         discrete_wheel_commands=presets.discrete_wheel_commands)
        publish = load_function(method(ROOT / 'rl_control/rl_control_node.py', 'publish_control_msg'), namespace)
        for action_id in (0, 1, 2):
            network, wheels = [], []
            node = types.SimpleNamespace(
                params={'discrete': 'T'}, dataset_logger=None, _discrete_action_id=action_id,
                get_clock=lambda: clock,
                pub_action_net=types.SimpleNamespace(publish=network.append),
                pub_action_motor=types.SimpleNamespace(publish=wheels.append))
            publish(node, [0.2, -0.1])
            self.assertEqual(len(network), 1)
            self.assertEqual(len(wheels), 1)
            self.assertEqual((wheels[0].vel_left, wheels[0].vel_right),
                             presets.discrete_wheel_commands(action_id))


if __name__ == '__main__':
    unittest.main()
