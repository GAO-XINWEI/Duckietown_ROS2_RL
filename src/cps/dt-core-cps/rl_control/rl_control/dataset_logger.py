"""Bounded asynchronous image/action recorder; no ROS or hardware dependencies."""
import csv
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path


BASE_FIELDS = ('frame_id', 'image', 'camera_timestamp_ns', 'action_timestamp_ns')


class DatasetLogger:
    """Drop newest on overflow; only the worker touches files and image codecs."""

    def __init__(self, directory, queue_size=32, max_image_bytes=4194304, metadata=None, action_mode='discrete'):
        if queue_size < 1:
            raise ValueError('queue_size must be positive')
        if action_mode not in ('discrete', 'continuous'):
            raise ValueError('action_mode must be discrete or continuous')
        fields = BASE_FIELDS + (('action_id',) if action_mode == 'discrete' else ('vel', 'angle'))
        if max_image_bytes < 1:
            raise ValueError('max_image_bytes must be positive')
        self.max_image_bytes = max_image_bytes
        self.queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.closed = threading.Event()
        self.stats = dict(submitted=0, written=0, dropped_full=0,
                          dropped_invalid=0, dropped_stopped=0, failed=0)
        self.error = None
        self.metadata = dict(metadata or {})
        self.metadata.update(schema_version=4, image_format='png', action_mode=action_mode,
                             image_stage='decoded_camera_before_resize',
                             max_image_bytes=max_image_bytes, queue_size=queue_size,
                             png_channels='RGB (decoded OpenCV BGR converted only for saving)')
        name = time.strftime('run_%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:12]
        self.run_dir = Path(directory).expanduser().resolve() / name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / 'images').mkdir()
        # Fail before starting control if the destination cannot be opened.
        self.csv_file = (self.run_dir / 'actions.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        try:
            self.writer.writeheader()
            self.csv_file.flush()
            self._write_metadata('recording')
        except Exception:
            self.csv_file.close()
            raise
        self.worker = threading.Thread(target=self._run, name='dataset-writer', daemon=True)
        self.worker.start()

    def submit(self, original_image, camera_timestamp_ns, publication):
        """Snapshot one successful decision. Never wait for disk or queue capacity."""
        with self.lock:
            frame_id = self.stats['submitted']
            self.stats['submitted'] += 1
            if self.stopping.is_set():
                self.stats['dropped_stopped'] += 1
                return False
            if (original_image.ndim != 3 or original_image.shape[2] != 3
                    or min(original_image.shape[:2]) < 1
                    or original_image.dtype.name != 'uint8'
                    or original_image.nbytes > self.max_image_bytes):
                self.stats['dropped_invalid'] += 1
                return False
            if self.queue.full():
                self.stats['dropped_full'] += 1
                return False
            row = dict(publication)
            row.update(frame_id=frame_id, camera_timestamp_ns=camera_timestamp_ns)
            try:
                self.queue.put_nowait((original_image.copy(), row))
            except queue.Full:
                self.stats['dropped_full'] += 1
                return False
            return True

    def snapshot(self):
        with self.lock:
            return dict(self.stats, error=self.error)

    def close(self, timeout=5.0):
        """Stop admission, drain in order, wait at most timeout; safe to repeat."""
        with self.lock:
            self.stopping.set()
        self.worker.join(timeout)
        return self.closed.is_set()

    def _write_metadata(self, status):
        info = dict(self.metadata, status=status, **self.snapshot())
        temp = self.run_dir / 'metadata.json.tmp'
        temp.write_text(json.dumps(info, indent=2) + '\n')
        os.replace(str(temp), str(self.run_dir / 'metadata.json'))

    def _save(self, payload, row):
        from PIL import Image
        # Original decoded HWC uint8 BGR; convert only the saved image to RGB.
        # No resizing, re-decoding or changes to the inference input.
        pixels = payload[:, :, ::-1]
        row['image'] = 'images/frame_{:09d}.png'.format(row['frame_id'])
        path = self.run_dir / row['image']
        temp = path.with_suffix(path.suffix + '.tmp')
        try:
            Image.fromarray(pixels).save(str(temp), format='PNG')
            os.replace(str(temp), str(path))
            self.writer.writerow(row)
            self.csv_file.flush()
        except Exception:
            for candidate in (temp, path):
                try:
                    candidate.unlink()
                except OSError:
                    pass
            raise

    def _run(self):
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    payload, row = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if self.error is None:
                        self._save(payload, row)
                        with self.lock:
                            self.stats['written'] += 1
                    else:
                        with self.lock:
                            self.stats['failed'] += 1
                except Exception as exc:
                    # Fail closed for logging, without throwing into motor control.
                    with self.lock:
                        self.error = '{}: {}'.format(type(exc).__name__, exc)
                        self.stats['failed'] += 1
                        self.stopping.set()
                finally:
                    self.queue.task_done()
        finally:
            try:
                self.csv_file.close()
                self._write_metadata('error' if self.error else 'closed')
            except Exception as exc:
                with self.lock:
                    self.error = '{}: {}'.format(type(exc).__name__, exc)
            finally:
                self.closed.set()
