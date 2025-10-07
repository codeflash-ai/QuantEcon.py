# This file is not meant for public use and will be removed v0.8.0.
# Use the `quantecon` namespace for importing the objects
# included below.

import warnings
from . import _filter

# Cache for DeprecationWarning messages to avoid repeated message construction
_warn_msg_cache: dict[str, str] = {}


__all__ = ['hamilton_filter']


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            "`quantecon.filter` is deprecated and has no attribute "
            f"'{name}'."
        )

    # Optimize warning message formatting
    msg = _warn_msg_cache.get(name)
    if msg is None:
        msg = (
            f"Please use `{name}` from the `quantecon` namespace, "
            "the `quantecon.filter` namespace is deprecated. You can use"
            f" the following instead:\n `from quantecon import {name}`."
        )
        _warn_msg_cache[name] = msg

    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)

    return getattr(_filter, name)
