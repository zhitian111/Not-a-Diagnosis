# utils/logger.py
import logging
import os
from datetime import datetime

def setup_logger(exp_name, log_dir, log_file_name):
    os.makedirs(log_dir, exist_ok=True)
    exp_dir = os.path.join(log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, log_file_name)

    logger = logging.getLogger(exp_name + "_" + log_file_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
