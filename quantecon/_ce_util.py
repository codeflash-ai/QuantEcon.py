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
    dtype = np.result_type(x1.dtype, x2.dtype)
    
    if x1.ndim == 1 and x2.ndim == 1:
        return _gridmake2_1d_1d(x1, x2, dtype)
    elif x1.ndim > 1 and x2.ndim == 1:
        return _gridmake2_nd_1d(x1, x2, dtype)
    else:
        raise NotImplementedError("Come back here")


@njit
def _gridmake2_1d_1d(x1, x2, dtype):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    out = np.empty((n1 * n2, 2), dtype=dtype)
    
    for i in range(n2):
        start_idx = i * n1
        end_idx = start_idx + n1
        out[start_idx:end_idx, 0] = x1
        out[start_idx:end_idx, 1] = x2[i]
    
    return out

@njit
def _gridmake2_nd_1d(x1, x2, dtype):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    ncols1 = x1.shape[1]
    out = np.empty((n1 * n2, ncols1 + 1), dtype=dtype)
    
    for i in range(n2):
        start_idx = i * n1
        end_idx = start_idx + n1
        out[start_idx:end_idx, :ncols1] = x1
        out[start_idx:end_idx, ncols1] = x2[i]
    
    return out
