"""
This file holds several functions that are used to solve matrix
equations.  Currently has functionality to solve:

* Lyapunov Equations
* Riccati Equations

TODO: 1. See issue 47 on github repository, should add support for
      Sylvester equations
      2. Fix warnings from checking conditioning of matrices
"""
import numpy as np
from numpy.linalg import solve
from scipy.linalg import solve_discrete_lyapunov as sp_solve_discrete_lyapunov
from scipy.linalg import solve_discrete_are as sp_solve_discrete_are
from numba import njit


EPS = np.finfo(float).eps


def solve_discrete_lyapunov(A, B, max_it=50, method="doubling"):
    r"""
    Computes the solution to the discrete lyapunov equation

    .. math::

        AXA' - X + B = 0

    :math:`X` is computed by using a doubling algorithm. In particular, we
    iterate to convergence on :math:`X_j` with the following recursions for
    :math:`j = 1, 2, \dots` starting from :math:`X_0 = B`, :math:`a_0 = A`:

    .. math::

        a_j = a_{j-1} a_{j-1}

    .. math::

        X_j = X_{j-1} + a_{j-1} X_{j-1} a_{j-1}'

    Parameters
    ----------
    A : array_like(float, ndim=2)
        An n x n matrix as described above.  We assume in order for
        convergence that the eigenvalues of A have moduli bounded by
        unity
    B : array_like(float, ndim=2)
        An n x n matrix as described above.  We assume in order for
        convergence that the eigenvalues of A have moduli bounded by
        unity
    max_it : scalar(int), optional(default=50)
        The maximum number of iterations
    method : string, optional(default="doubling")
        Describes the solution method to use.  If it is "doubling" then
        uses the doubling algorithm to solve, if it is "bartels-stewart"
        then it uses scipy's implementation of the Bartels-Stewart
        approach.

    Returns
    -------
    gamma1: array_like(float, ndim=2)
        Represents the value :math:`X`

    """
    if method == "doubling":
        A, B = list(map(np.atleast_2d, [A, B]))
        alpha0 = A
        gamma0 = B

        diff = 5
        n_its = 1

        while diff > 1e-15:

            alpha1 = alpha0 @ alpha0
            gamma1 = gamma0 + (alpha0 @ gamma0 @ alpha0.conjugate().T)

            diff = np.max(np.abs(gamma1 - gamma0))
            alpha0 = alpha1
            gamma0 = gamma1

            n_its += 1

            if n_its > max_it:
                msg = "Exceeded maximum iterations {}, check input matrics"
                raise ValueError(msg.format(n_its))

    elif method == "bartels-stewart":
        gamma1 = sp_solve_discrete_lyapunov(A, B)

    else:
        msg = "Check your method input. Should be doubling or bartels-stewart"
        raise ValueError(msg)

    return gamma1


def solve_discrete_riccati(A, B, Q, R, N=None, tolerance=1e-10, max_iter=500,
                           method="doubling"):
    """
    Solves the discrete-time algebraic Riccati equation

    .. math::

        X = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q

    Computation is via a modified structured doubling algorithm, an
    explanation of which can be found in the reference below, if
    `method="doubling"` (default), and via a QZ decomposition method by
    calling `scipy.linalg.solve_discrete_are` if `method="qz"`.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        k x k array.
    B : array_like(float, ndim=2)
        k x n array
    Q : array_like(float, ndim=2)
        k x k, should be symmetric and non-negative definite
    R : array_like(float, ndim=2)
        n x n, should be symmetric and positive definite
    N : array_like(float, ndim=2)
        n x k array
    tolerance : scalar(float), optional(default=1e-10)
        The tolerance level for convergence
    max_iter : scalar(int), optional(default=500)
        The maximum number of iterations allowed
    method : string, optional(default="doubling")
        Describes the solution method to use.  If it is "doubling" then
        uses the doubling algorithm to solve, if it is "qz" then it uses
        `scipy.linalg.solve_discrete_are` (in which case `tolerance` and
        `max_iter` are irrelevant).

    Returns
    -------
    X : array_like(float, ndim=2)
        The fixed point of the Riccati equation; a k x k array
        representing the approximate solution

    References
    ----------
    Chiang, Chun-Yueh, Hung-Yuan Fan, and Wen-Wei Lin. "STRUCTURED DOUBLING
    ALGORITHM FOR DISCRETE-TIME ALGEBRAIC RICCATI EQUATIONS WITH SINGULAR
    CONTROL WEIGHTING MATRICES." Taiwanese Journal of Mathematics 14, no. 3A
    (2010): pp-935.

    """
    methods = ['doubling', 'qz']
    if method not in methods:
        msg = "Check your method input. Should be {} or {}".format(*methods)
        raise ValueError(msg)

    # == Make sure that all array_likes are np arrays, two-dimensional == #
    A, B, Q, R = np.atleast_2d(A, B, Q, R)
    n, k = R.shape[0], Q.shape[0]
    I = np.identity(k)
    if N is None:
        N = np.zeros((n, k))
    else:
        N = np.atleast_2d(N)

    if method == 'qz':
        X = sp_solve_discrete_are(A, B, Q, R, e=I, s=N.T)
        return X
    
    # Ensure all arrays are float type for numba compatibility
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    Q = Q.astype(np.float64)
    R = R.astype(np.float64)
    N = N.astype(np.float64)
    I = I.astype(np.float64)
    
    candidates = (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0, 10e5)
    BB = B.T @ B
    BTA = B.T @ A
    best_gamma, R_hat, Q_tilde, G0, A0, H0 = _select_gamma_and_initialize(
        B, A, Q, R, N, I, BB, BTA, candidates
    )
    if best_gamma < 0.0:
        msg = "Unable to initialize routine due to ill conditioned arguments"
        raise ValueError(msg)

    gamma = best_gamma

    # == Main loop == #
    H1, converged, iterations = _doubling_iteration(I, A0, G0, H0, tolerance, max_iter)
    
    if not converged:
        fail_msg = "Convergence failed after {} iterations."
        raise ValueError(fail_msg.format(iterations))

    return H1 + gamma * I  # Return X


def solve_discrete_riccati_system(Π, As, Bs, Cs, Qs, Rs, Ns, beta,
                                  tolerance=1e-10, max_iter=1000):
    """
    Solves the stacked system of algebraic matrix Riccati equations
    in the Markov Jump linear quadratic control problems, by iterating
    Ps matrices until convergence.

    Parameters
    ----------
    Π : array_like(float, ndim=2)
        The Markov chain transition matrix with dimension m x m.
    As : array_like(float)
        Consists of m state transition matrices A(s) with dimension
        n x n for each Markov state s
    Bs : array_like(float)
        Consists of m state transition matrices B(s) with dimension
        n x k for each Markov state s
    Cs : array_like(float), optional(default=None)
        Consists of m state transition matrices C(s) with dimension
        n x j for each Markov state s. If the model is deterministic
        then Cs should take default value of None
    Qs : array_like(float)
        Consists of m symmetric and non-negative definite payoff
        matrices Q(s) with dimension k x k that corresponds with
        the control variable u for each Markov state s
    Rs : array_like(float)
        Consists of m symmetric and non-negative definite payoff
        matrices R(s) with dimension n x n that corresponds with
        the state variable x for each Markov state s
    Ns : array_like(float), optional(default=None)
        Consists of m cross product term matrices N(s) with dimension
        k x n for each Markov state,
    beta : scalar(float), optional(default=1)
        beta is the discount parameter
    tolerance : scalar(float), optional(default=1e-10)
        The tolerance level for convergence
    max_iter : scalar(int), optional(default=500)
        The maximum number of iterations allowed

    Returns
    -------
    Ps : array_like(float, ndim=2)
        The fixed point of the stacked system of algebraic matrix
        Riccati equations, consists of m n x n P(s) matrices

    """
    m = Qs.shape[0]
    k, n = Qs.shape[1], Rs.shape[1]
    # Create the Ps matrices, initialize as identity matrix
    Ps = np.array([np.eye(n) for i in range(m)])
    Ps1 = np.copy(Ps)

    # == Set up for iteration on Riccati equations system == #
    error = tolerance + 1
    fail_msg = "Convergence failed after {} iterations."

    # == Prepare array for iteration == #
    sum1, sum2 = np.empty((n, n)), np.empty((n, n))

    # == Main loop == #
    iteration = 0
    while error > tolerance:

        if iteration > max_iter:
            raise ValueError(fail_msg.format(max_iter))

        else:
            error = 0
            for i in range(m):
                # Initialize arrays
                sum1[:, :] = 0.
                sum2[:, :] = 0.
                for j in range(m):
                    sum1 += beta * Π[i, j] * As[i].T @ Ps[j] @ As[i]
                    sum2 += Π[i, j] * \
                            (beta * As[i].T @ Ps[j] @ Bs[i] + Ns[i].T) @ \
                            solve(Qs[i] + beta * Bs[i].T @ Ps[j] @ Bs[i],
                                  beta * Bs[i].T @ Ps[j] @ As[i] + Ns[i])

                Ps1[i][:, :] = Rs[i] + sum1 - sum2
                error += np.max(np.abs(Ps1[i] - Ps[i]))

            Ps[:, :, :] = Ps1[:, :, :]
            iteration += 1

    return Ps

@njit(cache=True)
def _select_gamma_and_initialize(
    B: np.ndarray, A: np.ndarray, Q: np.ndarray, R: np.ndarray, N: np.ndarray,
    I: np.ndarray, BB: np.ndarray, BTA: np.ndarray, candidates: tuple
):
    """
    Numba helper to select gamma and initialize matrices.
    Returns (gamma, R_hat, Q_tilde, G0, A0, H0)
    """
    current_min = np.inf
    best_gamma = -1.0
    n, k = R.shape[0], Q.shape[0]
    
    # Ensure B.T is float type for numba compatibility
    BT = B.T.astype(np.float64)
    
    for idx in range(len(candidates)):
        gamma = candidates[idx]
        Z = R + gamma * BB
        cn = np.linalg.cond(Z)
        if cn * EPS < 1:
            Z_invN = solve(Z, N + gamma * BTA)
            Q_tilde = -Q + (N.T @ Z_invN) + gamma * I

            Z_invBT = solve(Z, BT)
            G0 = B @ Z_invBT
            Z_invN_only = solve(Z, N)
            A0 = (I - gamma * G0) @ A - (B @ Z_invN_only)
            H0 = gamma * (A.T @ A0) - Q_tilde
            f1 = np.linalg.cond(Z, np.inf)
            f2 = gamma * f1
            G0H0 = G0 @ H0
            f3 = np.linalg.cond(I + G0H0)
            f_gamma = max(f1, f2, f3)
            if f_gamma < current_min:
                best_gamma = gamma
                R_hat = Z
                best_Q_tilde = Q_tilde
                best_G0 = G0
                best_A0 = A0
                best_H0 = H0
                current_min = f_gamma
    # Return values
    if current_min == np.inf:
        # Fail signal: gamma is negative
        return -1.0, None, None, None, None, None
    else:
        return best_gamma, R_hat, best_Q_tilde, best_G0, best_A0, best_H0

@njit(cache=True)
def _doubling_iteration(
    I: np.ndarray,
    A0: np.ndarray,
    G0: np.ndarray,
    H0: np.ndarray,
    tolerance: float,
    max_iter: int
):
    """
    Numba main loop for doubling algorithm.
    Returns H1 after convergence (A0, G0, H0 updated).
    Returns (H1, converged, iterations)
    """
    error = tolerance + 1
    i = 1
    while error > tolerance:
        if i > max_iter:
            # Return flag indicating convergence failure
            return H0, False, i
        # == Main iteration body == #
        IG0H0 = I + (G0 @ H0)
        IG0H0_invA0 = solve(IG0H0, A0)
        A1 = A0 @ IG0H0_invA0
        IH0G0 = I + (H0 @ G0)
        IH0G0_invA0T = solve(IH0G0, A0.T)
        G1 = G0 + ((A0 @ G0) @ IH0G0_invA0T)
        H0G0 = H0 @ G0
        IH0G0_invH0A0 = solve(IH0G0, H0 @ A0)
        H1 = H0 + (A0.T @ IH0G0_invH0A0)

        error = np.max(np.abs(H1 - H0))
        A0 = A1
        G0 = G1
        H0 = H1
        i += 1
    return H1, True, i
