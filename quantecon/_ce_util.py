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
            # Build entire cartesian product in one pass to avoid repeated tiling
            # of already-large intermediate matrices.
            # Preserve original behavior for d < 2 (which intentionally raises
            # IndexError in the original implementation when d == 1).
            # This vectorized approach produces the same row ordering:
            # first array varies fastest, last array varies slowest.
            lengths = [a.shape[0] for a in arrays]
            # If d == 0 or d == 1 the original implementation would have raised
            # via indexing; preserve that behavior by following the same control
            # flow (we already handled d == 2 above).
            out = None
            if d >= 3:
                total = int(np.prod(lengths))
                # Use a common dtype consistent with numpy stacking behavior
                out_dtype = np.result_type(*arrays)
                out = np.empty((total, d), dtype=out_dtype)
                cumulative = 1
                # For column k, each element of arrays[k] should be repeated 'rep'
                # times where rep = product(lengths[0:k]) and the whole pattern
                # is tiled 'cycles' times where cycles = total / (rep * len(arr_k)).
                for k, arr in enumerate(arrays):
                    rep = cumulative
                    cumulative *= arr.shape[0]
                    cycles = total // (rep * arr.shape[0])
                    # np.repeat then np.tile are implemented in C and efficient.
                    col = np.repeat(arr, rep)
                    if cycles > 1:
                        col = np.tile(col, cycles)
                    out[:, k] = col
            else:
                # This branch will not be reached for d == 2 (handled above).
                # Keep behavior consistent with original code path for d == 0 or 1:
                out = _gridmake2(arrays[0], arrays[1])


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
    if x1.ndim == 1 and x2.ndim == 1:
        # Allocate once and fill columns to avoid extra temporaries from column_stack
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        out_dtype = np.result_type(x1, x2)
        out = np.empty((n1 * n2, 2), dtype=out_dtype)
        out[:, 0] = np.tile(x1, n2)
        out[:, 1] = np.repeat(x2, n1)
        return out
    elif x1.ndim > 1 and x2.ndim == 1:
        # When x1 is 2D (previous grid) and x2 is 1D, build the tiled first
        # block by block to a single preallocated array and fill the last column
        # with repeated x2 values. This avoids creating full temporaries via
        # column_stack and reduces memory churn.
        n_rows_x1 = x1.shape[0]
        n_cols_x1 = x1.shape[1]
        m = x2.shape[0]
        total_rows = n_rows_x1 * m
        out_dtype = np.result_type(x1, x2)
        out = np.empty((total_rows, n_cols_x1 + 1), dtype=out_dtype)
        # Fill the repeated blocks of x1
        # Use a simple loop to copy blocks; this avoids building an intermediate
        # oversized tile at once. The copy cost is unavoidable, but we avoid
        # additional temporaries.
        for i in range(m):
            start = i * n_rows_x1
            end = start + n_rows_x1
            out[start:end, :n_cols_x1] = x1
        # Fill the last column with repeated x2 values (inner repeats = n_rows_x1)
        out[:, n_cols_x1] = np.repeat(x2, n_rows_x1)
        return out
    else:
        raise NotImplementedError("Come back here")
