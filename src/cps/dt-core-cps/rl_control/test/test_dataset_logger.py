"""Hardware-free tests: python3 -m unittest discover -s test -p test_dataset_logger.py."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import types
import numpy as np
from PIL import Image
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dataset_logger', ROOT / 'rl_control/dataset_logger.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DatasetLogger = module.DatasetLogger
JPEG = b'\xff\xd8\xfforiginal camera bytes'
INPUT = (np.arange(3 * 30 * 40, dtype=np.float32) % 256).reshape(3, 30, 40)
PUBLICATION = dict(vel=0.2, angle=-0.1, action_timestamp_ns=200)


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def logger(self, **kwargs):
        logger = DatasetLogger(self.directory.name, **kwargs)
        self.addCleanup(logger.close)
        return logger

    def test_pairing_bytes_timestamps_and_unique_run(self):
        logger = self.logger()
        for i in range(3):
            self.assertTrue(logger.submit(INPUT, 100 + i, dict(PUBLICATION, vel=i)))
        self.assertTrue(logger.close())
        with (logger.run_dir / 'actions.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        for i, row in enumerate(rows):
            with Image.open(logger.run_dir / row['image']) as image:
                self.assertEqual(image.size, (40, 30))
                np.testing.assert_array_equal(np.asarray(image).transpose(2, 0, 1), INPUT)
            self.assertEqual(int(row['camera_timestamp_ns']), 100 + i)
            self.assertEqual(float(row['vel']), i)
            self.assertEqual(int(row['action_timestamp_ns']), 200)
        info = json.loads((logger.run_dir / 'metadata.json').read_text())
        self.assertEqual(info['written'], 3)
        self.assertEqual(info['status'], 'closed')
        self.assertNotEqual(logger.run_dir, self.logger().run_dir)
        self.assertFalse(logger.submit(INPUT, 0, PUBLICATION))
        self.assertTrue(logger.close())

    def test_overflow_and_bounded_shutdown(self):
        logger = self.logger(queue_size=1)
        entered, release = threading.Event(), threading.Event()
        original = logger._save
        def slow_save(*args):
            entered.set()
            release.wait(3)
            original(*args)
        with patch.object(logger, '_save', slow_save):
            try:
                logger.submit(INPUT, 1, PUBLICATION)
                self.assertTrue(entered.wait(1))
                self.assertTrue(logger.submit(INPUT, 2, PUBLICATION))
                self.assertFalse(logger.submit(INPUT, 3, PUBLICATION))
                self.assertFalse(logger.close(timeout=0.01))
            finally:
                release.set()
                self.assertTrue(logger.close())
        self.assertEqual(logger.snapshot()['written'], 2)
        self.assertEqual(logger.snapshot()['dropped_full'], 1)

    def test_disk_failure_does_not_escape_or_leave_image(self):
        logger = self.logger()
        with patch.object(logger.writer, 'writerow', side_effect=OSError('disk full')):
            logger.submit(INPUT, 1, PUBLICATION)
            self.assertTrue(logger.close())
        self.assertEqual(list((logger.run_dir / 'images').iterdir()), [])
        self.assertEqual(logger.snapshot()['failed'], 1)
        self.assertIn('disk full', logger.snapshot()['error'])
        self.assertFalse(logger.submit(INPUT, 2, PUBLICATION))

    def test_png_encoding_runs_on_writer_thread_and_owns_snapshot(self):
        logger = self.logger()
        threads, entered, release = [], threading.Event(), threading.Event()
        original = Image.Image.save
        def save(image, *args, **kwargs):
            threads.append(threading.current_thread().name)
            entered.set()
            release.wait(3)
            return original(image, *args, **kwargs)
        source = INPUT.copy()
        with patch.object(Image.Image, 'save', save):
            try:
                self.assertTrue(logger.submit(source, 1, PUBLICATION))
                source[:] = 0
                self.assertTrue(entered.wait(1))
            finally:
                release.set()
                self.assertTrue(logger.close())
        self.assertEqual(threads, ['dataset-writer'])
        with Image.open(logger.run_dir / 'images/frame_000000000.png') as image:
            np.testing.assert_array_equal(np.asarray(image).transpose(2, 0, 1), INPUT)

    def test_invalid_shape(self):
        logger = self.logger()
        self.assertFalse(logger.submit(np.zeros((3, 480, 640)), 1, PUBLICATION))
        self.assertEqual(logger.snapshot()['dropped_invalid'], 1)
        with self.assertRaises(ValueError):
            self.logger(queue_size=0)


def node_methods():
    # Execute only methods with fake publishers; never import ROS, Torch or GPIO.
    tree = ast.parse((ROOT / 'rl_control/rl_control_node.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


class ControlTests(unittest.TestCase):
    def test_publication_values_and_callback_pairing(self):
        methods = node_methods()
        class Message:
            def __init__(self):
                self.header = types.SimpleNamespace(stamp=None)
        published = []
        class Publisher:
            def publish(self, msg):
                published.append(msg)
        times = iter([200, 201])
        class Clock:
            def now(self):
                ns = next(times)
                return types.SimpleNamespace(to_msg=lambda: types.SimpleNamespace(sec=0, nanosec=ns))
        constants = dict(GAIN=0.5, TRIM=0.0, K=27.0, WHEEL_DIST=0.102,
                         RADIUS=0.0318, LIMIT=1.0)
        namespace = dict(constants, ActionFloat32=Message, ActionFloat32Array2=Message,
                         WheelsCmdStamped=Message)
        exec(compile(ast.Module(body=[methods['publish_control_msg']], type_ignores=[]), '<test>', 'exec'), namespace)
        fake = types.SimpleNamespace(get_clock=lambda: Clock(), pub_action_net=Publisher(),
                                     pub_action_motor=Publisher())
        result = namespace['publish_control_msg'](fake, [0.2, -0.1])
        self.assertEqual(len(published), 2)
        self.assertEqual(published[0].data[0].data, [0.2])
        self.assertEqual(set(result), {'vel', 'angle', 'action_timestamp_ns'})
        self.assertAlmostEqual(published[1].vel_left, (0.2*0.8+0.5*0.1*0.102*4)/0.0318*0.5/27)
        self.assertEqual(result['action_timestamp_ns'], 200)
        class Array:
            @staticmethod
            def fromstring(value, dtype):
                return value
            uint8 = object()
        dict_ns = dict(np=Array)
        exec(compile(ast.Module(body=[methods['_img_callback']], type_ignores=[]), '<test>', 'exec'), dict_ns)
        records = []
        fake.params = {'discrete': 'F'}
        fake.get_logger = lambda: types.SimpleNamespace(info=lambda _: None)
        tensor = types.SimpleNamespace()
        tensor.detach = lambda: tensor
        tensor.cpu = lambda: tensor
        tensor.numpy = lambda: INPUT[None]
        fake.preprocess_img = lambda image: tensor
        fake.infer_action_continuous = lambda image: [0.2, -0.1]
        fake.publish_control_msg = lambda action: result
        fake.dataset_logger = types.SimpleNamespace(submit=lambda *args: records.append(args) or True)
        data = types.SimpleNamespace(data=JPEG, header=types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=1, nanosec=23)))
        dict_ns['_img_callback'](fake, data)
        self.assertEqual(len(records), 1)
        np.testing.assert_array_equal(records[0][0], INPUT)
        self.assertEqual(records[0][1:], (1000000023, result))
        fake.dataset_logger = None
        dict_ns['_img_callback'](fake, data)
        self.assertEqual(len(records), 1)
        fake.dataset_logger = types.SimpleNamespace(submit=lambda *args: records.append(args) or True)
        def fail(action):
            raise RuntimeError('publisher failed')
        fake.publish_control_msg = fail
        with self.assertRaises(RuntimeError):
            dict_ns['_img_callback'](fake, data)
        self.assertEqual(len(records), 1)


if __name__ == '__main__':
    unittest.main()
