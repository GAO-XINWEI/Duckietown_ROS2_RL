import os
import logging

from .constants import PERMISSIONS_DIR

# create logger
logging.basicConfig()
logger = logging.getLogger(os.environ.get('DT_MODULE_TYPE', 'module') + '.permissions')
logger.setLevel(logging.INFO)
if 'DEBUG' in os.environ and os.environ['DEBUG'].lower() in ['true', 'yes', '1']:
    logger.setLevel(logging.DEBUG)


def permission_granted(permission: str, strict: bool = False) -> bool:
    permission_file = os.path.join(PERMISSIONS_DIR, permission)
    # check if the permission file exists
    if not os.path.isfile(permission_file):
        if strict:
            raise FileNotFoundError(f"File '{permission_file}' not found.")
        return False
    # read permission file
    with open(permission_file, 'rt') as fin:
        permission_value = fin.read().strip()
    # parse value
    return permission_value.lower() in ['true', 'yes', '1']
