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
          'vel', 'angle')


class DatasetLogger:
    """Drop newest on overflow; only the worker touches files and image codecs."""

    def __init__(self, directory, queue_size=32, state_size=(30, 40), metadata=None):
        if queue_size < 1:
            raise ValueError('queue_size must be positive')
        self.input_shape = (3, int(state_size[0]), int(state_size[1]))
        if min(self.input_shape) < 1:
            raise ValueError('state_size must be positive')
        self.queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.closed = threading.Event()
        self.stats = dict(submitted=0, written=0, dropped_full=0,
                          dropped_invalid=0, dropped_stopped=0, failed=0)
        self.error = None
        self.metadata = dict(metadata or {})
        self.metadata.update(schema_version=2, image_format='png',
                             input_shape=list(self.input_shape), queue_size=queue_size,
                             png_channels='network channel 0, 1, 2 stored as R, G, B')
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

    def submit(self, network_input, camera_timestamp_ns, publication):
        """Snapshot one successful decision. Never wait for disk or queue capacity."""
        with self.lock:
            frame_id = self.stats['submitted']
            self.stats['submitted'] += 1
            if self.stopping.is_set():
                self.stats['dropped_stopped'] += 1
                return False
            if tuple(network_input.shape) != self.input_shape:
                self.stats['dropped_invalid'] += 1
                return False
            row = dict(publication)
            row.update(frame_id=frame_id, camera_timestamp_ns=camera_timestamp_ns)
            try:
                self.queue.put_nowait((network_input.copy(), row))
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
        import numpy as np
        from PIL import Image
        # Snapshot is the exact CPU CHW float input used by inference (0..255).
        # Keep channel identity; never re-run preprocessing or swap colors.
        pixels = np.rint(payload).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
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
