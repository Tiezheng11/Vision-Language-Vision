# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import warnings
from functools import lru_cache, wraps
from typing import Callable, Optional, TypeVar, Tuple
import torch
from torch import distributed as dist

T = TypeVar("T", bound=type)


def get_world_size_and_rank() -> Tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current process in the default process group.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if dist.is_available() and dist.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    else:
        return 1, 0

def get_logger(level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with a stream handler.

    Args:
        level (Optional[str]): The logging level. See https://docs.python.org/3/library/logging.html#levels for list of levels.

    Example:
        >>> logger = get_logger("INFO")
        >>> logger.info("Hello world!")
        INFO:torchtune.utils._logging:Hello world!

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    if level is not None:
        level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger


@lru_cache(None)
def log_once(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only once. LRU cache is used to ensure a specific message is
    logged only once, similar to how :func:`~warnings.warn` works when the ``once``
    rule is set via command-line or environment variable.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    log_rank_zero(logger=logger, msg=msg, level=level)


def deprecated(msg: str = "") -> Callable[[T], T]:
    """
    Decorator to mark an object as deprecated and print additional message.

    Args:
        msg (str): additional information to print after warning.

    Returns:
        Callable[[T], T]: the decorated object.
    """

    @lru_cache(maxsize=1)
    def warn(obj):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        warnings.warn(
            f"{obj.__name__} is deprecated and will be removed in future versions. "
            + msg,
            category=FutureWarning,
            stacklevel=3,
        )

    def decorator(obj):
        @wraps(obj)
        def wrapper(*args, **kwargs):
            warn(obj)
            return obj(*args, **kwargs)

        return wrapper

    return decorator


def log_rank_zero(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only on rank zero.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    logger.log(level, msg, stacklevel=2)


def deprecate_parameter(param_name: str, msg: str = "") -> Callable[[T], T]:
    """
    Decorator to mark a parameter as deprecated and print additional message.

    Args:
        param_name (str): The name of the parameter.
        msg (str): additional information to print after warning.

    Returns:
        Callable[[T], T]: the decorated object.
    """

    @lru_cache(maxsize=1)
    def warn(obj):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        warnings.warn(
            f"{param_name} is deprecated for {obj.__name__} and will be removed in future versions. "
            + msg,
            category=FutureWarning,
            stacklevel=3,
        )

    def decorator(obj):
        sig = inspect.signature(obj)

        @wraps(obj)
        def wrapper(*args, **kwargs):
            # Check positional and kwargs
            bound_args = sig.bind_partial(*args, **kwargs)
            all_args = {**bound_args.arguments}
            all_args.update(kwargs)
            if param_name in all_args:
                warn(obj)
            return obj(*args, **kwargs)

        return wrapper

    return decorator