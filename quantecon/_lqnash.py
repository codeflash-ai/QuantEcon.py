import numpy as np
from scipy.linalg import solve


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
    params = tuple(map(np.asarray, params))
    A, B1, B2, R1, R2, Q1, Q2, S1, S2, W1, W2, M1, M2 = params

    # == Multiply A, B1, B2 by sqrt(beta) to enforce discounting == #
    A, B1, B2 = [np.sqrt(beta) * x for x in (A, B1, B2)]

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


    # Precompute transposes that are reused
    B1T = B1.T
    B2T = B2.T
    W1T = W1.T
    W2T = W2.T
    M1T = M1.T
    M2T = M2.T

    v1 = np.eye(k_1)
    v2 = np.eye(k_2)
    P1 = np.zeros((n, n))
    P2 = np.zeros((n, n))
    F1 = np.full((k_1, n), np.inf)
    F2 = np.full((k_2, n), np.inf)

    for it in range(max_iter):
        # update
        F10 = F1
        F20 = F2

        # Solve small k x k systems for G1, G2 (inverse-like)
        S2mat = (B2T @ P2 @ B2) + Q2
        S1mat = (B1T @ P1 @ B1) + Q1
        G2 = solve(S2mat, v2)
        G1 = solve(S1mat, v1)

        # Compute H terms with a cheaper multiplication order
        # H1 = G1 @ (B1.T @ P1)
        B1T_P1 = B1T @ P1
        B2T_P2 = B2T @ P2
        H1 = G1 @ B1T_P1
        H2 = G2 @ B2T_P2

        # Reusable intermediate products
        H1_B2 = H1 @ B2
        H2_B1 = H2 @ B1
        G1_M1T = G1.dot(M1T)
        G2_M2T = G2.dot(M2T)
        H1_B2_plus = H1_B2 + G1_M1T
        H2_B1_plus = H2_B1 + G2_M2T

        H1_A = H1 @ A
        H2_A = H2 @ A
        G1_W1T = G1.dot(W1T)
        G2_W2T = G2.dot(W2T)
        H2_A_plus = H2_A + G2_W2T
        H1_A_plus = H1_A + G1_W1T

        # break up the computation of F1, F2
        F1_left = v1 - (H1_B2_plus @ H2_B1_plus)
        F1_right = H1_A_plus - (H1_B2_plus @ H2_A_plus)
        F1 = solve(F1_left, F1_right)
        F2 = H2_A + G2_W2T - (H2_B1_plus @ F1)

        # Update Lambdas

        Lambda1 = A - B2 @ F2
        Lambda2 = A - B1 @ F1
        Pi1 = R1 + (F2.T @ S1.dot(F2))
        Pi2 = R2 + (F1.T @ S2.dot(F1))

        # Update P1 using temporaries to avoid repeated computations
        Lambda1T = Lambda1.T
        Lambda2T = Lambda2.T

        LT_P1 = Lambda1T @ P1
        LT_P2 = Lambda2T @ P2

        LT_P1_L = LT_P1 @ Lambda1
        LT_P2_L = LT_P2 @ Lambda2

        LT_P1_B1 = LT_P1 @ B1
        LT_P2_B2 = LT_P2 @ B2

        F2T_M1 = F2.T.dot(M1)
        F1T_M2 = F1.T.dot(M2)

        # ((Lambda1.T @ P1 @ B1) + W1 - F2.T.dot(M1)) @ F1
        term1 = (LT_P1_B1 + W1 - F2T_M1) @ F1
        term2 = (LT_P2_B2 + W2 - F1T_M2) @ F2

        P1 = LT_P1_L + Pi1 - term1
        P2 = LT_P2_L + Pi2 - term2


        dd = np.max(np.abs(F10 - F1)) + np.max(np.abs(F20 - F2))

        if dd < tol:  # success!
            break

    else:
        msg = 'No convergence: Iteration limit of {0} reached in nnash'
        raise ValueError(msg.format(max_iter))

    return F1, F2, P1, P2
