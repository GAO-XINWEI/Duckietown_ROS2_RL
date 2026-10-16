"""Bounded asynchronous image/action recorder; no ROS or hardware dependencies."""
import csv
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path


FIELDS = ('frame_id', 'image', 'camera_timestamp_ns', 'action_timestamp_ns',
          'wheel_timestamp_ns', 'vel', 'angle', 'left', 'right')


class DatasetLogger:
    """Drop newest on overflow; only the worker touches files and image codecs."""

    def __init__(self, directory, queue_size=32, image_format='original',
                 max_image_bytes=4194304, metadata=None):
        if queue_size < 1 or max_image_bytes < 1:
            raise ValueError('queue_size and max_image_bytes must be positive')
        if image_format not in ('original', 'png'):
            raise ValueError('image_format must be original or png')
        self.image_format = image_format
        self.max_image_bytes = max_image_bytes
        self.queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.closed = threading.Event()
        self.stats = dict(submitted=0, written=0, dropped_full=0,
                          dropped_invalid=0, dropped_stopped=0, failed=0)
        self.error = None
        self.metadata = dict(metadata or {})
        self.metadata.update(schema_version=1, image_format=image_format,
                             queue_size=queue_size, max_image_bytes=max_image_bytes)
        name = time.strftime('run_%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:12]
        self.run_dir = Path(directory).expanduser().resolve() / name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / 'images').mkdir()
        # Fail before starting control if the destination cannot be opened.
        self.csv_file = (self.run_dir / 'actions.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=FIELDS)
        try:
            self.writer.writeheader()
            self.csv_file.flush()
            self._write_metadata('recording')
        except Exception:
            self.csv_file.close()
            raise
        self.worker = threading.Thread(target=self._run, name='dataset-writer', daemon=True)
        self.worker.start()

    def submit(self, image_bytes, camera_timestamp_ns, publication):
        """Snapshot one successful decision. Never wait for disk or queue capacity."""
        with self.lock:
            frame_id = self.stats['submitted']
            self.stats['submitted'] += 1
            if self.stopping.is_set():
                self.stats['dropped_stopped'] += 1
                return False
            if not image_bytes or len(image_bytes) > self.max_image_bytes:
                self.stats['dropped_invalid'] += 1
                return False
            row = dict(publication)
            row.update(frame_id=frame_id, camera_timestamp_ns=camera_timestamp_ns)
            try:
                self.queue.put_nowait((bytes(image_bytes), row))
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
        if self.image_format == 'png':
            import cv2
            import numpy as np
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError('Cannot decode camera image')
            ok, encoded = cv2.imencode('.png', decoded)
            if not ok:
                raise ValueError('Cannot encode PNG')
            payload, extension = encoded.tobytes(), 'png'
        elif payload.startswith(b'\xff\xd8\xff'):
            extension = 'jpg'
        elif payload.startswith(b'\x89PNG\r\n\x1a\n'):
            extension = 'png'
        else:
            raise ValueError('Unsupported compressed image signature')
        row['image'] = 'images/frame_{:09d}.{}'.format(row['frame_id'], extension)
        path = self.run_dir / row['image']
        temp = path.with_suffix(path.suffix + '.tmp')
        try:
            temp.write_bytes(payload)
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
