# Discrete fixed PWM presets + pre-Resize PNG dataset

This upload variant deliberately changes discrete wheel control:

| Raw action_id | Left / right target PWM | Both wheels |
| --- | --- | --- |
| 0 (left) | 230 / 255 | Forward |
| 1 (right) | 255 / 230 | Forward |
| 2 (straight) | 240 / 240 | Forward |

`discrete_pwm.py` defines presets and compensates for floor quantization. The
controller captures the raw ID regardless of dataset_enabled, keeps the existing
network/action publication and overrides the discrete wheel commands with these
presets. The `rl/action_cmd` vel/angle values still describe the original policy
conversion, not an inverse model of the overridden wheel commands. Dataset records
raw action_id. Inference/preprocessing are unchanged. Driver minimum PWM remains
120 and maximum changes to 255 on both sides. This driver change affects ALL
publishers, including continuous control; zero commands still map to zero PWM.
Stop other wheel-command publishers and retain emergency-stop priority.

This variant requires the matching controller, discrete_pwm.py and motor driver
files together. Updating just the controller against a MAX=200 driver produces
the wrong PWM values. It is packaged separately from the logger-only PR.

Eleven offline tests pass (eight dataset tests plus three PWM tests), using the
actual driver mapping and publication methods without importing hardware drivers.
No ROS2 Foxy/Jetson hardware validation has been performed. 240/255 are high
outputs; test with wheels lifted before ground operation. Equal PWM does not
ensure equal wheel speed, and these turn presets do not guarantee lane following.

## Dataset behavior

Recording is disabled by default. Each saved record contains only the PNG of the
decoded camera image before Resize, the corresponding raw discrete ACNet
`action_id` (0, 1 or 2), and
camera/action timestamps. It is paired inside the same camera callback, after
inference and both existing publications succeed. No timestamp matching is used.
Wheel commands are not recorded. PNGs retain the decoded camera dimensions.

## Enable on ROS2 Foxy / Jetson Nano

Build in your existing workspace with the normal model/ROS dependencies:

```sh
colcon build --packages-select rl_control --symlink-install
. install/setup.sh
ros2 launch rl_control rl_control_node.launch.xml \
  veh:=safeduckie3 discrete:=T model:=checkpoint_discrete_zigzag.pkl dataset_enabled:=true \
  dataset_directory:=/home/duckie/duckietown_dataset
```

Select the discrete checkpoint matching your experiment (the command above is an
example), and keep your existing model_path/device settings. The original launch
defaults are unchanged, so explicitly pass `discrete:=T` for your discrete model. Set
`dataset_enabled:=false` to disable all dataset work. Startup-only settings:

| Parameter / launch argument | Default | Meaning |
| --- | --- | --- |
| `dataset_enabled` | `false` | Enable recording |
| `dataset_directory` | `/home/duckie/duckietown_dataset` in launch; `~/duckietown_dataset` directly | Parent of independent UTC + random run directories |
| `dataset_queue_size` | `32` | Pending records; drop newest whole record when full |
| `dataset_max_image_bytes` | `4194304` | Maximum bytes per decoded uint8 BGR image; oversized records are dropped |

## What the PNG represents

The saved PNG is the uint8 camera image immediately after
`cv2.imdecode(..., cv2.IMREAD_COLOR)`, **before** `ToPILImage`, `Resize(STATE_SIZE)`
and `ToTensor`. The image is decoded only once; that same decoded frame goes
through the original preprocessing to produce the network input.

Saved width/height match the decoded camera image (for example, 640x480 if that is
what the camera publishes). They are no longer fixed at 40x30. The network still
receives its original 40x30 preprocessed tensor; saving does not change it.

The worker converts the saved copy from OpenCV BGR to RGB for a normal-color PNG.
PNG is lossless relative to the decoded camera pixels; it cannot recover detail
already lost in the camera's upstream JPEG encoding. There is no second JPEG
encoding or resizing in the writer.

To reproduce the existing network preprocessing later, load this PNG using
OpenCV (`cv2.imread`, which returns BGR) and apply the existing `trans_state`.
When loading with Pillow as RGB, convert to BGR before applying that pipeline.
This preserves the original controller's BGR-to-ToPILImage convention rather than
silently changing its colors. Metadata marks schema version 4 and
`image_stage=decoded_camera_before_resize` to distinguish these runs from older
40x30 network-input recordings.

## Output

```text
run_<UTC>_<unique>/
  images/frame_000000000.png
  actions.csv
  metadata.json
```

CSV columns:

```text
frame_id,image,camera_timestamp_ns,action_timestamp_ns,action_id
```

- `image`: relative path to that decision's full-size pre-Resize camera PNG.
- `camera_timestamp_ns`: original camera `header.stamp`, in integer ns.
- `action_timestamp_ns`: the existing action message header stamp, set from the
  node's ROS clock immediately before publishing `rl/action_cmd`.
- `action_id`: the exact 0/1/2 output returned by `ACNet_discrete.forward()` for
  this input, captured before conversion to control values. The network selects
  this ID using argmax of its three outputs. No second inference is performed.
  The original controller then maps the ID to `(vel, angle)` for driving; those
  converted values are not recorded in discrete mode.

For compatibility, if you explicitly run the original continuous mode
(`discrete:=F`), CSV uses `vel,angle` instead of `action_id`. The metadata records
which action schema applies. Discrete recordings contain only `action_id`.

`frame_id` is the submission sequence; gaps expose dropped records.

The action timestamp describes publication, not physical motor execution or DDS
receipt. Camera/node clocks may differ; zero/repeated camera stamps are preserved.
Metadata stores settings, saved image stage/channel convention and recording counters.
No motor constants or wheel data are stored. `Parameters.py` was inspected and
remains unchanged, as do preprocessing and network inference. Discrete wheel
commands and driver maximum are intentionally changed as described above.

## Queue, failures and stop

The callback copies the decoded pre-Resize uint8 image only after successful
inference/publication. PNG encoding and disk I/O run in a bounded background
queue. At 640x480, 32 queued BGR images occupy about 28.1 MiB, plus one worker
image and encoding/copy overhead. At the 4 MiB per-image limit, pending image
payloads can use up to 128 MiB. Reduce `dataset_queue_size` on Nano if needed.
Full queues are rejected before copying; submission never waits for capacity.
Python scheduling/copy/locks still cost time. PNGs are larger and encoding is
slower than for 40x30 inputs; verify timing, memory and dropped counters on Nano.

On overflow the whole newest record is dropped and counted. Wrong image
shape/dtype or decoded size above `dataset_max_image_bytes` is counted as
`dropped_invalid`. Only successfully saved decisions get an image/CSV row;
no row is emitted when inference/publication fails.

Counters in final metadata: `submitted`, `written`, `dropped_full`,
`dropped_invalid`, `dropped_stopped`, `failed`. After draining, submitted equals
the sum of written and dropped/failed records. Disk/encoding failure stops
recording, counts unsaved queued records and leaves control running. Startup
recording errors disable the logger with a ROS error message. A first rejected
submission warns; shutdown prints the final counters/error.

On Ctrl-C, destroy ROS resources first and drain recording for at most five
seconds. Slow/hung storage cannot indefinitely block exit. `closed` or `error`
metadata is finalized by the writer after draining; `recording` at exit means an
incomplete run. Image rename precedes CSV append/flush. Power/process loss can
leave orphan images, a partial last row or lost pending data; files are not a
transactional database and flush is not fsync. Validate incomplete runs before use.

## Verification

With NumPy and Pillow available (already used by the existing torchvision stack):

```sh
python3 -m unittest discover \
  -s src/cps/dt-core-cps/rl_control/test -p test_dataset_logger.py -v
```

Eight hardware-free tests check exact decoded-camera PNG round-trip including
BGR/RGB conversion and original dimensions, single-decode pre-Resize capture, timestamps/action pairing, snapshot ownership, background
encoding, unique runs, overflow, bounded/repeated stop, invalid input shape/dtype/size,
disk/CSV failure, all three raw discrete IDs before conversion, unchanged
intermediate policy conversion values and no record after failed publication. Fake publishers avoid
importing ROS, Torch or GPIO. Static comparison verifies trans_state, both
inference methods and wheel conversion are unchanged from the previous version.
Preprocessing adds an optional return of the same decoded frame, with no extra
decode or change to the original state calculation.

Before recording on Nano, verify with lifted wheels and an emergency stop, or
isolated replay with no motor-driver subscriber. The existing controller publishes
real motor commands. This logger does not touch GPIO or add a motor-stop guarantee;
use the existing safety controller/emergency stop before shutting down the node.

Run enabled/disabled with identical model settings and compare timing/behavior.
After safe stop, require metadata `closed`, check `written` equals CSV row count,
and check every CSV image exists with the decoded camera dimensions and a unique
frame ID. Replay known inputs and compare cv2.imread(PNG) pixel-for-pixel with
cv2.imdecode of the corresponding camera message,
and CSV action_id to the raw network output before the controller conversion.
Check action_timestamp_ns against `rl/action_cmd.header.stamp`. Stress `dataset_queue_size:=1` with slow
storage: expect whole-record drops and ID gaps, never shifted image/action pairs.
ROS2 Foxy/Jetson hardware timing is not validated by the offline tests.
