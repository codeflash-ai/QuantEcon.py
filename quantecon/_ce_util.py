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
    # Handle empty arrays case to match original behavior
    if len(arrays) == 0:
        return reduce(np.kron, arrays)
    # Fast path for njit
    arr_seq = tuple(arrays)
    return ckron_numba(arr_seq)


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
        arr_tuple = tuple(arrays)
        return gridmake_numba(arr_tuple)
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


@njit(cache=True, fastmath=True)
def _gridmake2_numba(x1, x2):
    # Copy of original _gridmake2 logic, numba-compatible
    if x1.ndim == 1 and x2.ndim == 1:
        # Create tiled and repeated arrays
        N1 = x1.shape[0]
        N2 = x2.shape[0]
        out = np.empty((N1*N2, 2), dtype=x1.dtype)
        for i in range(N2):
            for j in range(N1):
                out[i*N1+j, 0] = x1[j]
                out[i*N1+j, 1] = x2[i]
        return out
    elif x1.ndim > 1 and x2.ndim == 1:
        N1 = x1.shape[0]
        N2 = x2.shape[0]
        M = x1.shape[1]
        first = np.empty((N1 * N2, M), dtype=x1.dtype)
        for i in range(N2):
            for j in range(N1):
                for k in range(M):
                    first[i*N1 + j, k] = x1[j, k]
        second = np.empty(N1 * N2, dtype=x2.dtype)
        for i in range(N2):
            for j in range(N1):
                second[i*N1 + j] = x2[i]
        out = np.empty((N1 * N2, M + 1), dtype=x1.dtype)
        for idx in range(N1 * N2):
            for k in range(M):
                out[idx, k] = first[idx, k]
            out[idx, M] = second[idx]
        return out
    else:
        raise NotImplementedError("Come back here")

@njit(cache=True, fastmath=True)
def ckron_numba(arrays):
    # Numba-compatible Kronecker product reduction
    result = arrays[0]
    for i in range(1, len(arrays)):
        result = np.kron(result, arrays[i])
    return result

@njit(cache=True, fastmath=True)
def gridmake_numba(arrays):
    # Numba-compatible gridmake logic for all-1D arrays
    d = len(arrays)
    # force arrays to contiguous and squeeze
    seq = []
    for i in range(d):
        a = arrays[i].ravel()
        seq.append(a)
    if d == 2:
        out = _gridmake2_numba(seq[0], seq[1])
    else:
        out = _gridmake2_numba(seq[0], seq[1])
        for arr in seq[2:]:
            out = _gridmake2_numba(out, arr)
    return out
