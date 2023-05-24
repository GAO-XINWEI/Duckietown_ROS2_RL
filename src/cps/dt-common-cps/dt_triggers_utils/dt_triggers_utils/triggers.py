import os
import logging

from .constants import TRIGGERS_DIR

# create logger
logging.basicConfig()
logger = logging.getLogger(os.environ.get('DT_MODULE_TYPE', 'module') + '.triggers')
logger.setLevel(logging.INFO)
if 'DEBUG' in os.environ and os.environ['DEBUG'].lower() in ['true', 'yes', '1']:
    logger.setLevel(logging.DEBUG)


def set_trigger(key: str, signal: str):
    trigger_socket = os.path.join(TRIGGERS_DIR, key, 'socket')
    # check if the trigger file exists
    if not os.path.isfile(trigger_socket):
        msg = f"Trigger '{key}' not found."
        logger.error(msg)
        raise FileNotFoundError(msg)
    # write trigger signal
    logger.info(f"Sending signal '{signal}' to trigger '{key}'")
    with open(trigger_socket, 'wt') as fout:
        fout.write(signal)
