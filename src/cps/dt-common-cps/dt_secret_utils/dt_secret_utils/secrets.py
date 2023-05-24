import os
import logging

from .constants import SECRETS_DIR

# create logger
logging.basicConfig()
logger = logging.getLogger(os.environ.get('DT_MODULE_TYPE', 'module') + '.secrets')
logger.setLevel(logging.INFO)
if 'DEBUG' in os.environ and os.environ['DEBUG'].lower() in ['true', 'yes', '1']:
    logger.setLevel(logging.DEBUG)


def get_secret(key: str) -> str:
    secret_file = os.path.join(SECRETS_DIR, key)
    # check if the secret file exists
    if not os.path.isfile(secret_file):
        raise FileNotFoundError(f"File '{secret_file}' not found.")
    # read secret file
    with open(secret_file, 'rt') as fin:
        secret_value = fin.read().strip()
    # return secret
    return secret_value
