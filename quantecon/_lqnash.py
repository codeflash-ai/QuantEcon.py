import numpy as np
from numba import njit


def nnash(A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2,
          beta=1.0, tol=1e-8, max_iter=1000):
    """
    Compute the limit of a Nash linear quadratic dynamic game. In this
    problem, player i minimizes

    .. math::
        \sum_{t=0}^{\infty}
        \left\{
            x_t' r_i x_t + 2 x_t' w_i
            u_{it} +u_{it}' q_i u_{it} + u_{jt}' s_i u_{jt} + 2 u_{jt}'
            m_i u_{it}
        \right\}

    subject to the law of motion

    .. math::
        x_{t+1} = A x_t + b_1 u_{1t} + b_2 u_{2t}

    and a perceived control law :math:`u_j(t) = - f_j x_t` for the other
    player.

    The solution computed in this routine is the :math:`f_i` and
    :math:`p_i` of the associated double optimal linear regulator
    problem.

    Parameters
    ----------
    A : array_like(float)
        Corresponds to the above equation, should be of size (n, n)
    B1 : scalar(float) or array_like(float)
        As above, size (n, k_1)
    B2 : scalar(float) or array_like(float)
        As above, size (n, k_2)
    R1 : scalar(float) or array_like(float)
        As above, size (n, n)
    R2 : scalar(float) or array_like(float)
        As above, size (n, n)
    Q1 : scalar(float) or array_like(float)
        As above, size (k_1, k_1)
    Q2 : scalar(float) or array_like(float)
        As above, size (k_2, k_2)
    S1 : scalar(float) or array_like(float)
        As above, size (k_1, k_1)
    S2 : scalar(float) or array_like(float)
        As above, size (k_2, k_2)
    W1 : scalar(float) or array_like(float)
        As above, size (n, k_1)
    W2 : scalar(float) or array_like(float)
        As above, size (n, k_2)
    M1 : scalar(float) or array_like(float)
        As above, size (k_2, k_1)
    M2 : scalar(float) or array_like(float)
        As above, size (k_1, k_2)
    beta : scalar(float), optional(default=1.0)
        Discount rate
    tol : scalar(float), optional(default=1e-8)
        This is the tolerance level for convergence
    max_iter : scalar(int), optional(default=1000)
        This is the maximum number of iteratiosn allowed

    Returns
    -------
    F1 : array_like, dtype=float, shape=(k_1, n)
        Feedback law for agent 1
    F2 : array_like, dtype=float, shape=(k_2, n)
        Feedback law for agent 2
    P1 : array_like, dtype=float, shape=(n, n)
        The steady-state solution to the associated discrete matrix
        Riccati equation for agent 1
    P2 : array_like, dtype=float, shape=(n, n)
        The steady-state solution to the associated discrete matrix
        Riccati equation for agent 2

    """
    # == Unload parameters and make sure everything is an array == #
    params = A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2
    params = list(map(np.asarray, params))
    A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2 = params

    n = A.shape[0]

    if B1.ndim == 1:
        k_1 = 1
        B1 = np.reshape(B1, (n, 1))
    else:
        k_1 = B1.shape[1]

    if B2.ndim == 1:
        k_2 = 1
        B2 = np.reshape(B2, (n, 1))
    else:
        k_2 = B2.shape[1]

    # Ensure correct types and contiguous arrays for numba
    A = np.ascontiguousarray(A, dtype=np.float64)
    B1 = np.ascontiguousarray(B1, dtype=np.float64)
    B2 = np.ascontiguousarray(B2, dtype=np.float64)
    R1 = np.ascontiguousarray(R1, dtype=np.float64)
    R2 = np.ascontiguousarray(R2, dtype=np.float64)
    Q1 = np.ascontiguousarray(Q1, dtype=np.float64)
    Q2 = np.ascontiguousarray(Q2, dtype=np.float64)
    S1 = np.ascontiguousarray(S1, dtype=np.float64)
    S2 = np.ascontiguousarray(S2, dtype=np.float64)
    W1 = np.ascontiguousarray(W1, dtype=np.float64)
    W2 = np.ascontiguousarray(W2, dtype=np.float64)
    M1 = np.ascontiguousarray(M1, dtype=np.float64)
    M2 = np.ascontiguousarray(M2, dtype=np.float64)

    F1, F2, P1, P2 = _nnash_core(A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2, beta, tol, max_iter)
    if F1 is None:
        msg = 'No convergence: Iteration limit of {0} reached in nnash'
        raise ValueError(msg.format(max_iter))
    return F1, F2, P1, P2


@njit(cache=True)
def _nnash_core(A: np.ndarray,
                B1: np.ndarray,
                B2: np.ndarray,
                R1: np.ndarray,
                R2: np.ndarray,
                Q1: np.ndarray,
                Q2: np.ndarray,
                S1: np.ndarray,
                S2: np.ndarray,
                W1: np.ndarray,
                W2: np.ndarray,
                M1: np.ndarray,
                M2: np.ndarray,
                beta: float,
                tol: float,
                max_iter: int):
    """
    Internal core for the Nash LQ solver. The outer nnash function handles
    ndarray conversion, shape normalization, and exception raising.
    """
    # == Multiply A, B1, B2 by sqrt(beta) to enforce discounting == #
    sqrt_beta = np.sqrt(beta)
    A = sqrt_beta * A
    B1 = sqrt_beta * B1
    B2 = sqrt_beta * B2

    n = A.shape[0]
    k_1 = B1.shape[1]
    k_2 = B2.shape[1]

    v1 = np.eye(k_1)
    v2 = np.eye(k_2)
    P1 = np.zeros((n, n))
    P2 = np.zeros((n, n))
    F1 = np.full((k_1, n), np.inf)
    F2 = np.full((k_2, n), np.inf)

    for it in range(max_iter):
        F10 = F1.copy()
        F20 = F2.copy()

        # Use np.linalg.solve in place of scipy.linalg.solve (Numba supported)
        G2 = np.linalg.solve((B2.T @ P2 @ B2) + Q2, v2)
        G1 = np.linalg.solve((B1.T @ P1 @ B1) + Q1, v1)
        H2 = G2 @ B2.T @ P2
        H1 = G1 @ B1.T @ P1

        H1B2 = H1 @ B2
        G1M1T = G1 @ M1.T
        H2B1 = H2 @ B1
        G2M2T = G2 @ M2.T

        F1_left = v1 - (H1B2 + G1M1T) @ (H2B1 + G2M2T)
        H1A = H1 @ A
        G1W1T = G1 @ W1.T
        H2A = H2 @ A
        G2W2T = G2 @ W2.T
        F1_right = H1A + G1W1T - (H1B2 + G1M1T) @ (H2A + G2W2T)

        F1 = np.linalg.solve(F1_left, F1_right)
        F2 = H2A + G2W2T - (H2B1 + G2M2T) @ F1

        Lambda1 = A - B2 @ F2
        Lambda2 = A - B1 @ F1
        Pi1 = R1 + (F2.T @ S1 @ F2)
        Pi2 = R2 + (F1.T @ S2 @ F1)

        Lambda1TP1B1 = Lambda1.T @ P1 @ B1
        Lambda2TP2B2 = Lambda2.T @ P2 @ B2

        F2TM1 = F2.T @ M1
        F1TM2 = F1.T @ M2

        P1 = (Lambda1.T @ P1 @ Lambda1) + Pi1 - ((Lambda1TP1B1 + W1 - F2TM1) @ F1)
        P2 = (Lambda2.T @ P2 @ Lambda2) + Pi2 - ((Lambda2TP2B2 + W2 - F1TM2) @ F2)

        dd = np.max(np.abs(F10 - F1)) + np.max(np.abs(F20 - F2))

        if dd < tol:
            break
    else:
        # Numba cannot raise Python exceptions use sentinel + outer raising
        return None, None, None, None

    return F1, F2, P1, P2
