#!/usr/bin/env python3

import os


def get_ood_dir() -> str:
    return os.environ.get('ROBOT_OOD_DIR')
