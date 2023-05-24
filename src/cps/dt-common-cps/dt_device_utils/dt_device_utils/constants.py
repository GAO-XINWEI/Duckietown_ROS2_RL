#!usr/bin/python3
from enum import IntEnum

CONFIG_DIR = "/data/config/"
DEVICE_ID_IFACE = "eth0"


class DeviceHardwareBrand(IntEnum):
    UNKNOWN = 0
    RASPBERRY_PI = 1
    JETSON_NANO = 2
    RASPBERRY_PI_64 = 3


class DeviceHardwareModel(IntEnum):
    UNKNOWN = 0
    # Raspberry Pi Family
    RASPBERRY_PI_2 = 1
    RASPBERRY_PI_3B = 2
    RASPBERRY_PI_3B_PLUS = 3
    RASPBERRY_PI_4B = 4
    # Jetson Nano Family
    JETSON_NANO = 11
    JETSON_NANO_2GB = 12
