# This file is not meant for public use and will be removed v0.8.0.
# Use the `quantecon` namespace for importing the objects
# included below.

import warnings
from . import _kalman


__all__ = ['Kalman']


def __dir__():
    return __all__


def __getattr__(name):
    # Fast-path: compare directly to avoid linear search if only a single item
    if name != 'Kalman':
        raise AttributeError(
                "`quantecon.kalman` is deprecated and has no attribute "
                f"'{name}'."
            )

    # Pre-build warning message for efficiency
    warnings.warn(
        "Please use `Kalman` from the `quantecon` namespace, the"
        "`quantecon.kalman` namespace is deprecated. You can use"
        " the following instead:\n `from quantecon import Kalman`.",
        category=DeprecationWarning, stacklevel=2
    )

    return getattr(_kalman, name)
