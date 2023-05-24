# ros2_video_streamer

> tldr: A ROS2 python node for streaming video files or images to a topic.

## Usage

There is a `ros2_video_streamer_node` launch file. Minimally, you need to provide
a `type` (e.g. video or image) and `path`, which is the full path to the content
to stream.

```bash
ros2 launch ros2_video_streamer ros2_video_streamer_node type:=<type> path:=<path>
```

The content is published on the `~/image/compressed` topic.

## Additional Settings

There are also the following optional settings that you can pass to the launch file

* __node_name__ - Override the default name of the node.

* __image_topic_name__ - Override default name of the topic to publish images to

* __info_topic_name__ - Override default name of the topic to publish camera info to

* __config_file_name__ - Name of YAML file in the `config` folder. `CameraInfo` messages are published on the `~/camera_info` topic based on the content of the config file. By default, nothing is published.

* __loop__ (_true_ or _false_) - Continously publish the source on loop

* __frame_id__ - Frame id string in the `CameraInfo` messages.

* __start__ (_int_) - Location to start publishing the source.
