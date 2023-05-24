import os
import time
import yaml

from typing import Any, Tuple, Optional


__all__ = [
    'save_calibration',
    'read_calibration'
]


BASE_DIR = os.environ.get("ROBOT_CALIBRATION_DIR", None)

if BASE_DIR is None:
    raise RuntimeError("No `ROBOT_CALIBRATION_DIR` environment variable \
defined.")


def save_calibration(rel_file_path: str, data) -> str:
    """Save data into YAML file
    Returns the full path to the written YAML file.
    """
    assert(rel_file_path.endswith(".yaml"))
    data["calibration_time"] = time.strftime("%d-%b-%Y %H-%M-%S")
    full_fp = os.path.join(BASE_DIR, rel_file_path)
    # If the file already exists, then rename it to something unique.
    if os.path.isfile(full_fp):
        # Get file name (excluding extensions)
        filename = os.path.basename(full_fp).split('.')[0]
        unique_val = time.strftime("%d-%b-%Y-%H-%M-%S")
        os.rename(src=full_fp,
            dst=os.path.join(os.path.dirname(full_fp),
            f"{filename}-{unique_val}-old.yaml"))

    try:
        os.makedirs(os.path.dirname(full_fp))
    except OSError as _:
        pass
    with open(full_fp, 'w') as outfile:
        outfile.write(yaml.safe_dump(data, default_flow_style=False))
    return full_fp

def read_calibration(rel_file_path: str) -> Tuple[Optional[Any], str]:
    """Read the calibration file and return its contents.

    :param rel_file_path: Relative file path to the YAML calibration file.
        For example, `rel_file_path` can be intrinsic/default.yaml.
    :type rel_file_path: str
    :return: The first element in the tuple contents the data in the YAML file.
        It will be `None` if unable to read in the data. The second element in
        the tuple is the full constructed file path to the YAML file.
    :rtype: Tuple[Optional[Any], str]
    """
    assert(rel_file_path.endswith('.yaml'))
    full_fp = os.path.join(BASE_DIR, rel_file_path)
    if not os.path.isfile(full_fp):
        return None, full_fp
    else:
        with open(full_fp, 'r') as in_file:
            try:
                data = yaml.safe_load(in_file)
                if "calibration_time" in data:
                    del data["calibration_time"]
                return data, full_fp
            except yaml.YAMLError as _:
                return None, full_fp
