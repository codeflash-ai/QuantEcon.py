"""
Utility functions used in CompEcon

Based routines found in the CompEcon toolbox by Miranda and Fackler.

References
----------
Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
and Finance, MIT Press, 2002.

"""
import numpy as np
from numba import njit


def ckron(*arrays):
    """
    Repeatedly applies the np.kron function to an arbitrary number of
    input arrays

    Parameters
    ----------
    *arrays : tuple/list of np.ndarray

    Returns
    -------
    out : np.ndarray
          The result of repeated kronecker products.

    Notes
    -----
    Based of original function `ckron` in CompEcon toolbox by Miranda
    and Fackler.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    return _ckron_impl(arrays)


def gridmake(*arrays):
    """
    Expands one or more vectors (or matrices) into a matrix where rows span the
    cartesian product of combinations of the input arrays. Each column of the
    input arrays will correspond to one column of the output matrix.

    Parameters
    ----------
    *arrays : tuple/list of np.ndarray
              Tuple/list of vectors to be expanded.

    Returns
    -------
    out : np.ndarray
          The cartesian product of combinations of the input arrays.

    Notes
    -----
    Based of original function ``gridmake`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
    and Finance, MIT Press, 2002.

    """
    if all([i.ndim == 1 for i in arrays]):
        return _gridmake_impl(arrays)
    else:
        raise NotImplementedError("Come back here")


def _gridmake2(x1, x2):
    """
    Expands two vectors (or matrices) into a matrix where rows span the
    cartesian product of combinations of the input arrays. Each column of the
    input arrays will correspond to one column of the output matrix.

    Parameters
    ----------
    x1 : np.ndarray
         First vector to be expanded.

    x2 : np.ndarray
         Second vector to be expanded.

    Returns
    -------
    out : np.ndarray
          The cartesian product of combinations of the input arrays.

    Notes
    -----
    Based of original function ``gridmake2`` in CompEcon toolbox by
    Miranda and Fackler.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
    and Finance, MIT Press, 2002.

    """
    if x1.ndim == 1 and x2.ndim == 1:
        return np.column_stack([np.tile(x1, x2.shape[0]),
                               np.repeat(x2, x1.shape[0])])
    elif x1.ndim > 1 and x2.ndim == 1:
        first = np.tile(x1, (x2.shape[0], 1))
        second = np.repeat(x2, x1.shape[0])
        return np.column_stack([first, second])
    else:
        raise NotImplementedError("Come back here")

@njit(cache=True)
def _ckron_impl(arrays):
    acc = arrays[0]
    for i in range(1, len(arrays)):
        acc = np.kron(acc, arrays[i])
    return acc

@njit(cache=True)
def _gridmake2_impl(x1, x2):
    if x1.ndim == 1 and x2.ndim == 1:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        result = np.empty((n1 * n2, 2), dtype=x1.dtype)
        for i in range(n2):
            for j in range(n1):
                result[i * n1 + j, 0] = x1[j]
                result[i * n1 + j, 1] = x2[i]
        return result
    elif x1.ndim > 1 and x2.ndim == 1:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        first = np.empty((n1 * n2, x1.shape[1]), dtype=x1.dtype)
        for i in range(n2):
            for j in range(n1):
                first[i * n1 + j, :] = x1[j, :]
        second = np.empty(n1 * n2, dtype=x2.dtype)
        for i in range(n2):
            for j in range(n1):
                second[i * n1 + j] = x2[i]
        return np.column_stack((first, second))
    else:
        raise NotImplementedError("Come back here")

@njit(cache=True)
def _gridmake_impl(arrays):
    d = len(arrays)
    if d == 2:
        out = _gridmake2_impl(arrays[0], arrays[1])
    else:
        out = _gridmake2_impl(arrays[0], arrays[1])
        for i in range(2, d):
            out = _gridmake2_impl(out, arrays[i])
    return out
