"""
Utility functions used in CompEcon

Based routines found in the CompEcon toolbox by Miranda and Fackler.

References
----------
Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
and Finance, MIT Press, 2002.

"""
from functools import reduce
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
    return reduce(np.kron, arrays)


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
        d = len(arrays)
        if d == 2:
            out = _gridmake2(*arrays)
        else:
            out = _gridmake2(arrays[0], arrays[1])
            for arr in arrays[2:]:
                out = _gridmake2(out, arr)

        return out
    else:
        raise NotImplementedError("Come back here")


@njit(cache=True, fastmath=True)
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
        m = x1.shape[0]
        n = x2.shape[0]
        temp = np.empty(1, dtype=x1.dtype) + np.empty(1, dtype=x2.dtype)
        result_dtype = temp.dtype
        out = np.empty((m * n, 2), dtype=result_dtype)
        for j in range(n):
            for i in range(m):
                out[j * m + i, 0] = x1[i]
                out[j * m + i, 1] = x2[j]
        return out
    elif x1.ndim > 1 and x2.ndim == 1:
        m = x1.shape[0]
        n = x2.shape[0]
        k = x1.shape[1]
        temp = np.empty(1, dtype=x1.dtype) + np.empty(1, dtype=x2.dtype)
        result_dtype = temp.dtype
        out = np.empty((m * n, k + 1), dtype=result_dtype)
        for j in range(n):
            for i in range(m):
                for col in range(k):
                    out[j * m + i, col] = x1[i, col]
                out[j * m + i, k] = x2[j]
        return out
    else:
        raise NotImplementedError("Come back here")
