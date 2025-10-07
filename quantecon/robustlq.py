# This file is not meant for public use and will be removed v0.8.0.
# Use the `quantecon` namespace for importing the objects
# included below.

import warnings
from . import _robustlq


__all__ = ['RBLQ']


def __dir__():
    return __all__


def __getattr__(name):
    if name != "RBLQ":
        raise AttributeError(
            "`quantecon.robustlq` is deprecated and has no attribute "
            f"'{name}'."
        )
    # Message construction merged for slightly faster performance
    warnings.warn(
        "Please use `RBLQ` from the `quantecon` namespace, the"
        " `quantecon.robustlq` namespace is deprecated. You can use"
        " the following instead:\n `from quantecon import RBLQ`.",
        category=DeprecationWarning, stacklevel=2
    )
    return getattr(_robustlq, "RBLQ")
