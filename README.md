# Quick Handbook for Duckietown
## Duckietown Resource
Official website: `https://duckietown.com`
Official Github: `https://github.com/duckietown` to access the ROS and other resource.
Gym Dukckietown: `https://github.com/duckietown/gym-duckietown` to access the RL training environment. 

## Image Install
Download the system image from the Lab NAS on location: `/Duckietown/JetonNanoDB21M-by-Xinwei/JetsonBackup.img.gz`. You can flash the SD Card following the command in `README.md` file.

The the system image already contain:
- Duckietown
- ROS2
- Pytorch

Now you can ssh and login with:
```
ssh duckie@192.168.0.172
(password)quackquack
```

## Command to Run

Open the Duckietown ROS package and build the ROS bag with:
```
cd ~/dt_ws
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/dt_ws/install/setup.bash
```

Run the Lane Following RL within two different terminal. The first command will launch the `rl_control_node`, which is loading the RL network:
```
ros2 launch rl_control rl_control_node.launch.xml
```

The second command will launch the Duckiebot standard underlaying service, like motor, camera and so on:
```
ros2 launch dt_demos rl_lane_following_a.launch.xml
```

### Debug cases: 
 1. The network is loaded but the camera node is fail with running time error: 
    - Make sure that the first command load with enough time, so that the camera node can work in limited time.
 2. The camera nodes is fail to launch due to the `GPIO`, please: 
	 - Try to restart the camera node and make sure you shutdown the camera node and the `GPIO` with `Ctrl + C` . See `~/dt_ws/src/cps/dt-duckiebot-interface-cps/camera_driver/camera_driver/jetson_nano_camera_node.py`for detail.
	 - Try to reboot the Duckiebot.
	 - Try to check the physical wire connection of the Camera device.

## Modify Lane Following Policy
You can access the RL policy to control at the folder at `~/dt_ws/src/cps/dt-core-cps/rl_control` and customize it according to your needs.
