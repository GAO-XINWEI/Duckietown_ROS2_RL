#!/usr/bin/env python3

import re
import time
import yaml
import serial
import semver
import traceback
import dataclasses

from logging import Logger
from threading import Thread
from typing import Callable, Optional
from serial.tools.list_ports import grep as serial_grep

from battery_drivers.constants import (
    BATTERY_PCB16_READY_VID,
    BATTERY_PCB16_READY_PID,
    BATTERY_PCB16_BAUD_RATE
)


KELVIN_TO_CELSIUS = lambda k: k - 273.15


@dataclasses.dataclass
class BatteryInteraction:
    name: str
    command: bytes
    check: Callable
    callback: Optional[Callable] = None
    answer: Optional[dict] = None
    active: bool = True

    def join(self):
        while True:
            if not self.active:
                return
            time.sleep(0.1)

    def complete(self, answer: Optional[dict]):
        self.answer = answer
        self.active = False
        if self.callback is not None:
            self.callback(self.answer)


class Battery:
    def __init__(self, callback, logger: Logger = None):
        self._devices = []
        self._info = None
        self._data = None
        self._device = None
        self._interaction: Optional[BatteryInteraction] = BatteryInteraction(
            name="get_info",
            command=b'??',
            check=lambda d: 'FirmwareVersion' in d,
            callback=lambda d: setattr(self, '_info', self._format_info(d))
        )
        self._is_shutdown = False
        self._logger = None
        self._logger = logger
        # if not callable(callback):
        #     raise ValueError('Callback must be a callable object.')
        self._callback = callback
        self._worker = Thread(target=self._work)

    def start(self, block: bool = False, quiet: bool = True):
        if block:
            return self._work(quiet=quiet)
        else:
            self._worker.start()

    def join(self):
        if self._worker.is_alive():
            self._worker.join()

    def is_shutdown(self):
        return self._is_shutdown

    def shutdown(self):
        #   This is NOT a battery shutdown, it simply shuts down the drivers
        self._is_shutdown = True
        self.join()

    def turn_off(self, timeout: int = 20, wait: bool = False,
        callback: Optional[Callable] = None):
        #   This is a battery shutdown, the power will be cut off
        #   after `timeout` seconds
        firmware_version = self.info.get("version")
        # noinspection PyBroadException
        try:
            # multi-firmware support
            if semver.compare(firmware_version, "2.0.0") == 0:
                timeout_str = f'{timeout}'.zfill(2)
                self._interaction = BatteryInteraction(
                    name="turn_off",
                    command=f'Q{timeout_str}'.encode('utf-8'),
                    check=lambda d: d.get('TTL(sec)', None) == timeout,
                    callback=callback,
                )
            elif semver.compare(firmware_version, "2.0.1") >= 0:
                self._interaction = BatteryInteraction(
                    name="turn_off",
                    command="QQ".encode('utf-8'),
                    check=lambda d: d.get('QACK', None) is not None,
                    callback=callback
                )
            else:
                raise Exception()
        except Exception:
            self._logger.warning(f"Unknown/Unsupported battery firmware:"\
                " {firmware_version}")
            pass

        if wait:
            self._interaction.join()

    @property
    def info(self):
        return self._info

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._chew_on_data()

    def _find_device(self):
        vid_pid_match = "VID:PID={}:{}".format(
            BATTERY_PCB16_READY_VID, BATTERY_PCB16_READY_PID)
        ports = serial_grep(vid_pid_match)
        self._devices = [p.device for p in ports]  # ['/dev/ttyACM0', ...]

    def _chew_on_data(self):
        if self._data and self._info:
            self._data = {
                "present": True,
                "charging": self._data['input_voltage'] >= (5.0 / 2),
                **self._data
            }

    def _read_next(self, dev, quiet: bool = False):
        try:
            raw = dev.read_until().decode('utf-8', 'ignore')
            cleaned = re.sub(r"\x00\s*", "", raw).rstrip()
            cleaned = re.sub(r"-\s+", "-", cleaned)
            try:
                # if "}{" present, get the first for checking shutdown ACK
                if "}{" in cleaned:
                    cleaned = cleaned.split("}{")[0] + "}"
                parsed = yaml.load(cleaned, yaml.SafeLoader)
                return parsed
            except yaml.YAMLError as e:
                if self._logger:
                    self._logger.error(str(e))
                return None
        except BaseException as e:
            if quiet:
                # do not print, it would swamp the logs
                pass
            else:
                raise e

    def _work(self, quiet: bool = True):
        while True:
            if self._is_shutdown:
                return
            # ---
            # if we don't have a battery device, search again
            if len(self._devices) == 0:
                self._find_device()
            # if we still don't have it, just sleep for 5 seconds
            if len(self._devices) == 0:
                self._logger.warning('No battery found. Retrying in 5 seconds.')
                pass
            else:
                # we have at least one candidate device, try reading
                for device in self._devices:
                    try:
                        self.read_battery(device, quiet)
                    except (PermissionError, serial.serialutil.SerialException):
                        self._logger.error(f"Unable to access {device}")
                        time.sleep(5)
            if self._logger:
                self._logger.warning('An error occurred while reading from the battery.')
            time.sleep(1)

    def read_battery(self, device, quiet):
        with serial.Serial(device, BATTERY_PCB16_BAUD_RATE) as dev:
            # once the device is open, try reading from it forever
            # break only on unknown errors
            while True:
                if self._is_shutdown:
                    return
                # ---
                if self._data is not None:
                    # we were able to read from the battery at least once,
                    # consume any pending interaction
                    if self._interaction.active:
                        iname, icmd = self._interaction.name, self._interaction.command
                        # there is a command to be sent, send it and continue
                        self._logger.info(f"Pending interaction '{iname}' found. "
                                            f"Sending {str(icmd)} to the battery")
                        dev.write(self._interaction.command)
                        dev.flush()
                # ---
                try:
                    parsed = self._read_next(dev, quiet=quiet)
                    if parsed is None:
                        continue
                    # first time we read?
                    if self._device is None:
                        self._device = device
                        self._logger.info('Battery found at {}.'.format(device))
                    # distinguish between 'data' packet and others
                    if 'SOC(%)' in parsed:
                        # this is a 'data' packet
                        self.data = self._format_data(parsed)
                    elif self._interaction.active:
                        iname = self._interaction.name
                        self._logger.info(f"Received (candidate) response to "
                                            f"interaction '{iname}': {str(parsed)}")
                        if self._interaction.check(parsed):
                            self._logger.info(f"Received valid response to "
                                                f"interaction '{iname}': {str(parsed)}")
                            # complete interaction
                            self._interaction.complete(parsed)
                except BaseException as e:
                    if quiet:
                        traceback.print_exc()
                        break
                    raise e

    @staticmethod
    def _format_data(data):
        return {
            "temperature": round(KELVIN_TO_CELSIUS(data['CellTemp(degK)']), 2),
            "cell_voltage": round(float(data['CellVoltage(mV)']) / 1000, 2),
            "input_voltage": round(float(data['ChargerVoltage(mV)']) / 1000, 2),
            "current": round(float(data['Current(mA)']) / 1000, 2),
            "cycle_count": data['CycleCount'],
            "percentage": data['SOC(%)'],
            "time_to_empty": int(data['TimeToEmpty(min)'] * 60),
            "usb_out_1_voltage": round(float(data['USB OUT-1(mV)']) / 1000, 2),
            "usb_out_2_voltage": round(float(data['USB OUT-2(mV)']) / 1000, 2)
        }

    @staticmethod
    def _format_info(info):
        boot_data = str(info["BootData"])
        firmware_version = str(info["FirmwareVersion"])
        major, minor, patch, *_ = firmware_version + "000"
        yy, mm, dd = boot_data[3:5], boot_data[5:7], boot_data[7:9]
        return {
            "version": f"{major}.{minor}.{patch}",
            "boot": {
                "version": boot_data[0],
                "pcb_version": boot_data[1:3],
                "date": f"{mm}/{dd}/{yy}"
            },
            "serial_number": info["SerialNumber"]
        }
