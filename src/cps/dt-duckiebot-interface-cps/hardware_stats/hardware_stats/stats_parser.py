#!/usb/bin/env python3

# https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3261/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/AppendixTegraStats.html#

import re
import subprocess

from typing import Dict
from dataclasses import dataclass


__all__ = ['RAM', 'IRAM', 'Swap', 'CPUs', 'GR3D_FREQ', 'EMC_FREQ',
    'Temperatures', 'PowerConsumption', 'Stats', 'TegraStatsParser']


@dataclass
class RAM:
    used: float = -1                    # MB
    total: float = -1                   # MB
    free_ram_blocks: int = 0
    size_free_ram_blocks: float = -1    # MB


@dataclass
class IRAM:
    used: float = -1    # kB
    total: float = -1   # kB
    size: float = -1    # kB


@dataclass
class Swap:
    used: float  = -1   # MB
    total: float  = -1  # MB
    cached: float = -1  # MB


@dataclass
class CPUs:
    # Number of CPU cores
    num: int = 0
    frequencies: Dict[str, float] = None    # Megahertz
    load: Dict[str, float] = None           # Percentage


@dataclass
class GR3D_FREQ:
    used: float = -1        # Percentage
    frequency: float = -1   # Megahertz


@dataclass
class EMC_FREQ:
    used: float = -1       # Percentage
    frequency: float = -1  # Megahertz


@dataclass
class Temperatures:
    nums: int = 0
    temps: Dict[str, float] = None # Celsius


@dataclass
class PowerConsumption:
    nums: int = 0
    # Instantaneous power consumption in milliwatts
    inst_pwr: Dict[str, float] = None 
    # Average power consumption in milliwatts
    avg_pwr: Dict[str, float] = None


@dataclass
class Stats:
    ram: RAM
    iram: IRAM
    swap: Swap
    cpus: CPUs
    gr3d: GR3D_FREQ
    emc: EMC_FREQ
    temps: Temperatures
    pwr: PowerConsumption



class TegraStatsParser:
    def __init__(self):
        pass

    def execute(self):
        cmd = ['tegrastats']
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line 
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    @staticmethod
    def parse_ram(line):
        data = re.findall(r'RAM ([0-9]*)\/([0-9]*)MB \(lfb ([0-9]*)x([0-9]*)MB\)', line)
        if len(data) == 0:
            return RAM()
        data = data[0]
        return RAM(
            used=float(data[0]),
            total=float(data[1]),
            free_ram_blocks=int(data[2]),
            size_free_ram_blocks=float(data[3]))

    @staticmethod
    def parse_swap(line):
        data = re.findall(r'SWAP ([0-9]*)\/([0-9]*)MB \(cached ([0-9]*)MB\)', line)
        if len(data) == 0:
            return Swap()
        data = data[0]
        return Swap(used=data[0], total=data[1], cached=data[2])

    @staticmethod
    def parse_iram(line):
        data = re.findall(r'IRAM ([0-9]*)\/([0-9]*)kB \(lfb ([0-9]*)kB\)', line)
        if len(data) == 0:
            return IRAM()
        data = data[0]
        return IRAM(used=data[0], total=data[1], size=data[2])

    @staticmethod
    def parse_cpus(line):
        data = re.findall(r'CPU \[(.*)\]', line)[0]
        freqs = re.findall(r'@([0-9]*)', data)
        loads = re.findall(r'([0-9]*)%', data)
        if len(freqs) == len(loads) == 0:
            return CPUs()
        return CPUs(num=len(freqs),
            frequencies={str(k):float(v) for k,v in enumerate(freqs)},
            load={str(k):float(v) for k,v in enumerate(loads)},)

    @staticmethod
    def parse_gr3d(line):
        data = re.findall(r'GR3D_FREQ ([0-9]*)%@?([0-9]*)?', line)
        if len(data) == 0:
            return GR3D_FREQ()
        data = data[0]
        if data[1] == '':
            return GR3D_FREQ(used=float(data[0]))
        else:
            return GR3D_FREQ(used=float(data[0]), frequency=float(data[1]))

    @staticmethod
    def parse_emc(line):
        data = re.findall(r'EMC_FREQ ([0-9]*)%@?([0-9]*)?', line)
        if len(data) == 0:
            return EMC_FREQ()
        data = data[0]
        if data[1] == '':
            return EMC_FREQ(used=float(data[0]))
        else:
            return EMC_FREQ(used=float(data[0]), frequency=float(data[1]))

    @staticmethod
    def parse_temperatures(line):
        data = re.findall(r'([A-Za-z]*)@([0-9.]*)C', line)
        if len(data) == 0:
            return Temperatures()
        return Temperatures(len(data), temps={k:float(v) for k,v in data})

    @staticmethod
    def parse_vdds(line):
        data = re.findall(r'VDD([A-Za-z]*)\s*([0-9]*)/([0-9]*)', line)
        if len(data) == 0:
            return PowerConsumption()
        return PowerConsumption(nums=len(data), 
            inst_pwr={k:v for k,v,_ in data},
            avg_pwr={k:v for k,_,v in data},)

    @staticmethod
    def parse_data(line):
        return Stats(ram=TegraStatsParser.parse_ram(line),
            iram=TegraStatsParser.parse_iram(line),
            swap=TegraStatsParser.parse_swap(line),
            cpus=TegraStatsParser.parse_cpus(line),
            gr3d=TegraStatsParser.parse_gr3d(line),
            emc=TegraStatsParser.parse_emc(line),
            temps=TegraStatsParser.parse_temperatures(line),
            pwr=TegraStatsParser.parse_vdds(line))


if __name__ == "__main__":
    print("*** Usage example ***")
    for line in TegraStatsParser().execute():
        print(TegraStatsParser.parse_data(line))
        print("-----------------------------")
