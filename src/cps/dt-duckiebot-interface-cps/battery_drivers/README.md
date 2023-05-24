# battery_drivers
All credits to ![Duckietown](https://github.com/duckietown/dt-device-health/tree/daffy/packages/battery_drivers).
A ROS2 package is added to publish the data on the battery on topics.

## Permissions
If you enter permissions error such as __'Unable to access /dev/ttyACM0'__, you need to add yourself into the `dialout` group:
```bash
sudo usermod -a -G dialout $USER
```
You may need to restart the device afterwards.
