# ACNet image/action dataset recording (ROS2 Foxy / Jetson Nano)

Recording is **disabled by default**. Each admitted record comes directly from one
camera callback: the exact compressed image used for inference, the returned
`(vel, angle)` and the actual published `(left, right)` wheel values. No temporal
matching is performed. Inference, preprocessing, discrete-action mapping, wheel
conversion, limits and publication order are unchanged. `Parameters.py` was read
and its motor constants are recorded in metadata; no training parameters change.

## Build and enable

In the existing Foxy workspace on the Nano, with its normal ROS and model dependencies:

```sh
colcon build --packages-select rl_control --symlink-install
. install/setup.sh
ros2 launch rl_control rl_control_node.launch.xml \
  veh:=safeduckie3 dataset_enabled:=true \
  dataset_directory:=/home/duckie/duckietown_dataset
```

Retain your existing `model_path`, `model`, `device`, `quantize` and `discrete`
launch arguments as appropriate. The default launch still starts the controller
with its existing model/vehicle defaults. Set `dataset_enabled:=false` to disable
recording, with no writer thread, dataset directory or extra image codec work.
Parameters are read at startup; restart to change them.

| Launch argument / ROS parameter | Default | Meaning |
| --- | --- | --- |
| `dataset_enabled` | `false` | Opt in to recording |
| `dataset_directory` | `/home/duckie/duckietown_dataset` in launch; `~/duckietown_dataset` directly | Parent directory; a unique UTC + random run directory is always created |
| `dataset_queue_size` | `32` | Maximum pending records; newest is dropped when full |
| `dataset_image_format` | `original` | Copy original JPEG/PNG bytes; `png` requests background decode and PNG encoding |
| `dataset_max_image_bytes` | `4194304` | Reject empty or oversized payloads before queueing |

Example for explicitly requested PNG: append `dataset_image_format:=png`.
Original JPEG is recommended on Nano: it preserves camera bytes exactly and
avoids re-encoding. Original mode detects JPEG/PNG by byte signature, independent
of the ROS `format` string. Other compression types stop recording with an error.
PNG mode uses the existing OpenCV/NumPy dependencies only in the writer thread.

## Output and timestamps

```text
run_20261016T120000Z_<unique>/
  images/frame_000000000.jpg
  actions.csv
  metadata.json
```

One CSV row per **successfully saved** decision, with columns:

- `frame_id`: callback submission sequence within the run (gaps expose drops).
- `image`: image path relative to the run directory.
- `camera_timestamp_ns`: unmodified `CompressedImage.header.stamp` as integer ns.
- `action_timestamp_ns`: the header timestamp of the published `rl/action_cmd`.
- `wheel_timestamp_ns`: the header timestamp of the published wheel command.
- `vel`, `angle`: ACNet control action returned by the existing inference method
  (including its existing clipping/discrete conversion); not wheel duty cycles.
- `left`, `right`: actual `WheelsCmdStamped.vel_left/vel_right` values after the
  existing motor conversion and limiting.

The two publication stamps use the node's ROS clock immediately before their
respective `publish` calls. They do not prove DDS delivery, physical motor
execution or synchronized camera/node clocks. Camera stamps are preserved even
if zero or repeated. Do not assume a fixed latency, a fixed frame rate, or that
adjacent rows are contiguous RL transitions/episodes when IDs have gaps.

Metadata includes model/vehicle settings, STATE_SIZE, motor constants, clock type,
queue configuration, completion status and counters. SPEED_FACTOR=0.8 and
ANGLE_FACTOR=4 remain in the original wheel conversion. Values in the CSV are
captured from the same published messages, not recomputed by the writer.

## Backpressure, errors and shutdown

The callback copies camera bytes (already needed for inference), then after both
publishes succeed submits a small record with `put_nowait`. It performs no disk
I/O or image encoding. Queue memory is bounded by queue size and payload limit
(32 x 4 MiB pending by default, plus one worker payload and transient copies;
PNG encoding also needs a decoded frame). Size this conservatively for Nano.
The queue does not block waiting for capacity. Python scheduling, byte copying
and lock contention still have cost; this is not a hard real-time guarantee.

If full, **drop the newest whole record**, not its image or action independently.
No row is produced for inference/publish exceptions or failed image writes. On
write/codec failure, the writer stops admission and counts queued unsaved records
as failed; control continues. A first rejected submission warns in the ROS log;
shutdown prints the counters/error. Startup path/parameter errors disable
recording with an error message while leaving the controller available.

Counters: `submitted`, `written`, `dropped_full`, `dropped_invalid`,
`dropped_stopped`, `failed`. After a normal drain, submitted equals written plus
all dropped/failed records. Each image is atomically renamed before its CSV row
is appended and flushed. On a write exception temporary/current image files are
removed where possible. CSV and image files are not a transactional database:
process/power loss may leave orphan images or a partial last CSV row. Flush is
not fsync; recordings are not guaranteed against power loss.

On Ctrl-C, ROS resources are destroyed first, then the writer stops accepting
records and drains for at most 5 seconds. Repeated close is safe. A slow/hung disk
may outlast this timeout; the daemon writer cannot block process exit. Such a run
can have lost pending records and metadata still marked `recording`. Final
`closed`/`error` metadata is written by the worker only after its drain. Treat
anything else as incomplete and validate files before training.

## Validation and hardware safety

Run the hardware-free tests from the repository root (Python 3.8+):

```sh
python3 -m unittest discover \
  -s src/cps/dt-core-cps/rl_control/test -p test_dataset_logger.py -v
```

They exercise byte preservation, direct image/action pairing and exact stamps,
unique directories, queue overflow, bounded/repeated close, oversized payloads,
invalid compression, disk/CSV failure, actual published wheel values, and no
record when publishing fails. They use fake publishers and never import ROS,
Torch, Jetson.GPIO or motor drivers. They do not establish real Nano timing.

On the Nano, first validate with wheels lifted and an available emergency stop,
or replay a short bag in an isolated ROS namespace with **no motor driver
subscriber**. The controller publishes real wheel commands when started; changing
the namespace alone is not a substitute for isolating/remapping hardware topics.
This change neither touches GPIO nor changes the original controller's shutdown
motor behavior. Do not rely on the logger or node exit as a motor stop; use your
existing safety controller/emergency stop.

1. Run once disabled and once enabled with the same existing launch/model settings.
   Confirm identical inference and motor behavior; compare callback timing and
   dropped counters under sustained load on the Nano.
2. Stop safely, then Ctrl-C. Require `metadata.json` status `closed`, and verify
   every CSV image exists, every image has a corresponding row, IDs are unique,
   and the written count equals CSV row count. Preserve ns columns as integers.
3. Replay known JPEG messages with unique header stamps and distinct pixel content;
   compare saved bytes to the original `CompressedImage.data` and CSV values to
   both observed published messages. Repeat for discrete/continuous modes you use.
4. Stress with `dataset_queue_size:=1` and PNG or slow storage. Expect whole-record
   drops and ID gaps, without shifted image/action pairing. Check memory/timing
   on the Nano before using recordings in an experiment.
5. Check startup with an unwritable destination and simulate full storage in an
   isolated test setup. Confirm recording reports failure and control remains
   responsive. Forced termination should be recognized as an incomplete run.

Retain ROS bags when raw topic history is required. This dataset contains only
camera images actually processed into successfully published decisions, subject
to explicit storage/backpressure losses.
