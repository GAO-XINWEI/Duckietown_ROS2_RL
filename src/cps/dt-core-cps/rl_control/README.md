# Save ACNet network-input PNG + action + timestamps

Recording is disabled by default. Each saved record contains only the PNG of the
actual preprocessed network input, the corresponding ACNet `(vel, angle)`, and
camera/action timestamps. It is paired inside the same camera callback, after
inference and both existing publications succeed. No timestamp matching is used.
Wheel commands and original full-resolution camera photos are not recorded.

## Enable on ROS2 Foxy / Jetson Nano

Build in your existing workspace with the normal model/ROS dependencies:

```sh
colcon build --packages-select rl_control --symlink-install
. install/setup.sh
ros2 launch rl_control rl_control_node.launch.xml \
  veh:=safeduckie3 dataset_enabled:=true \
  dataset_directory:=/home/duckie/duckietown_dataset
```

Keep your existing model/device/discrete launch arguments. Set
`dataset_enabled:=false` to disable all dataset work. Startup-only settings:

| Parameter / launch argument | Default | Meaning |
| --- | --- | --- |
| `dataset_enabled` | `false` | Enable recording |
| `dataset_directory` | `/home/duckie/duckietown_dataset` in launch; `~/duckietown_dataset` directly | Parent of independent UTC + random run directories |
| `dataset_queue_size` | `32` | Pending records; drop newest whole record when full |

## What the PNG represents

The logger snapshots `preprocessed_img`, the same CPU tensor passed into the
existing inference method (which copies it to the inference device). It does
not decode/resize the camera image a second time. Current `STATE_SIZE=(30,40)`
produces a PNG **40 pixels wide and 30 pixels high**.

The original pipeline produces a 0..255 float32 CHW tensor from uint8 resized
pixels. In the background worker, values are rounded to the nearest uint8 and
saved losslessly as PNG. This recovers the pixel levels; PNG does not retain tiny
float32 arithmetic differences. To reconstruct the input, read PNG as RGB,
transpose HWC to CHW, cast to float32 and add the batch dimension, without another
resize, channel swap or multiplication by 255.

Channel identity is preserved: network channels 0/1/2 are PNG R/G/B. The original
code feeds OpenCV BGR into ToPILImage without swapping colors, so these channels
correspond to camera B/G/R. A normal PNG viewer can therefore show swapped colors
relative to the camera photo. This deliberately preserves what the network saw;
the controller's preprocessing and inference are unchanged. When loading with
OpenCV, convert its default BGR decode to RGB before reconstructing the tensor.

## Output

```text
run_<UTC>_<unique>/
  images/frame_000000000.png
  actions.csv
  metadata.json
```

CSV columns:

```text
frame_id,image,camera_timestamp_ns,action_timestamp_ns,vel,angle
```

- `image`: relative path to that decision's network-input PNG.
- `camera_timestamp_ns`: original camera `header.stamp`, in integer ns.
- `action_timestamp_ns`: the existing action message header stamp, set from the
  node's ROS clock immediately before publishing `rl/action_cmd`.
- `vel,angle`: the existing inference method's returned control action (including
  its existing clipping/discrete conversion).
- `frame_id`: submission sequence; gaps expose dropped records.

The action timestamp describes publication, not physical motor execution or DDS
receipt. Camera/node clocks may differ; zero/repeated camera stamps are preserved.
Metadata stores settings, input shape/channel convention and recording counters.
No motor constants or wheel data are stored. `Parameters.py` was inspected and
remains unchanged, as do preprocessing, inference, wheel conversion and limits.

## Queue, failures and stop

Only the small network input is copied in the callback. PNG encoding and disk I/O
run in a bounded background queue. With the current input shape, 32 queued
float32 snapshots require about 450 KiB, plus worker/encoding overhead. Queue
submission never waits for capacity; Python scheduling/copy/locks still cost time.
On overflow the whole newest record is dropped and counted. An admitted record
gets one image and one CSV row; no row is emitted when inference/publication fails.

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

Six hardware-free tests check exact network-pixel PNG round-trip including channel
order and dimensions, timestamps/action pairing, snapshot ownership, background
encoding, unique runs, overflow, bounded/repeated stop, invalid input shape,
disk/CSV failure and no record after failed publication. Fake publishers avoid
importing ROS, Torch or GPIO. Static comparison also verifies the original
preprocessing, inference and wheel conversion methods are unchanged.

Before recording on Nano, verify with lifted wheels and an emergency stop, or
isolated replay with no motor-driver subscriber. The existing controller publishes
real motor commands. This logger does not touch GPIO or add a motor-stop guarantee;
use the existing safety controller/emergency stop before shutting down the node.

Run enabled/disabled with identical model settings and compare timing/behavior.
After safe stop, require metadata `closed`, check `written` equals CSV row count,
and check every CSV image exists with size 40x30 and a unique frame ID. Replay
known inputs and compare PNG pixels to the rounded tensor passed to inference,
and CSV action/stamp to `rl/action_cmd`. Stress `dataset_queue_size:=1` with slow
storage: expect whole-record drops and ID gaps, never shifted image/action pairs.
ROS2 Foxy/Jetson hardware timing is not validated by the offline tests.
