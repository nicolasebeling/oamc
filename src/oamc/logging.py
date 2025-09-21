"""Logging utilities for the OAMC package."""

import logging
import logging.config
import os
from datetime import datetime
from typing import Literal

import numpy
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def enable_logging(
    console_level: Literal[0, 10, 20, 30, 40, 50] = logging.INFO,
    file_level: Literal[0, 10, 20, 30, 40, 50] = logging.DEBUG,
    directory: str = "logs",
):
    os.makedirs(directory, exist_ok=True)

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file = os.path.join(directory, f"{time}.log")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s.%(msecs)03d %(levelname)s from %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": file,
                "mode": "w",
                "formatter": "default",
                "level": file_level,
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": console_level,
            },
        },
        "loggers": {
            "oamc": {
                "handlers": ["file", "console"],
                "level": logging.DEBUG,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": logging.DEBUG,
        },
    }

    logging.config.dictConfig(config)


def format_array_for_logging(array: NDArray) -> str:
    return numpy.array2string(
        array,
        max_line_width=1000,
        formatter={"float_kind": lambda x: "% .3f" % x},
    ).replace("\n", "")
