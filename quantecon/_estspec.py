"""
Functions for working with periodograms of scalar data.

"""
import numpy as np
from numba import njit
from numpy.fft import fft


def smooth(x, window_len=7, window='hanning'):
    """
    Smooth the data in x using convolution with a window of requested
    size and type.

    Parameters
    ----------
    x : array_like(float)
        A flat NumPy array containing the data to smooth
    window_len : scalar(int), optional
        An odd integer giving the length of the window.  Defaults to 7.
    window : string
        A string giving the window type. Possible values are 'flat',
        'hanning', 'hamming', 'bartlett' or 'blackman'

    Returns
    -------
    array_like(float)
        The smoothed values

    Notes
    -----
    Application of the smoothing window at the top and bottom of x is
    done by reflecting x around these points to extend it sufficiently
    in each direction.

    """
    if len(x) < window_len:
        raise ValueError("Input vector length must be >= window length.")

    if window_len < 3:
        raise ValueError("Window length must be at least 3.")

    if not window_len % 2:  # window_len is even
        window_len += 1
        print("Window length reset to {}".format(window_len))

    # === Select window values === #
    win_fn = None
    if window == 'hanning':
        win_fn = _hanning_window
    elif window == 'hamming':
        win_fn = _hamming_window
    elif window == 'bartlett':
        win_fn = _bartlett_window
    elif window == 'blackman':
        win_fn = _blackman_window
    elif window == 'flat':
        win_fn = _flat_window
    else:
        msg = "Unrecognized window type '{}'".format(window)
        print(msg + " Defaulting to hanning")
        win_fn = _hanning_window

    s = _compute_reflected_signal(x, window_len)
    w = win_fn(window_len)
    return _convolve_valid(w, s)


def periodogram(x, window=None, window_len=7):
    """
    Computes the periodogram

    .. math::

        I(w) = \frac{1}{n} \Big[ \sum_{t=0}^{n-1} x_t e^{itw} \Big] ^2

    at the Fourier frequencies :math:`w_j := \frac{2 \pi j}{n}`,
    :math:`j = 0, \dots, n - 1`, using the fast Fourier transform. Only the
    frequencies :math:`w_j` in :math:`[0, \pi]` and corresponding values
    :math:`I(w_j)` are returned. If a window type is given then smoothing
    is performed.

    Parameters
    ----------
    x : array_like(float)
        A flat NumPy array containing the data to smooth
    window_len : scalar(int), optional(default=7)
        An odd integer giving the length of the window.  Defaults to 7.
    window : string
        A string giving the window type. Possible values are 'flat',
        'hanning', 'hamming', 'bartlett' or 'blackman'

    Returns
    -------
    w : array_like(float)
        Fourier frequencies at which periodogram is evaluated
    I_w : array_like(float)
        Values of periodogram at the Fourier frequencies

    """
    n = len(x)
    I_w = np.abs(fft(x))**2 / n
    w = 2 * np.pi * np.arange(n) / n  # Fourier frequencies
    w, I_w = w[:int(n/2)+1], I_w[:int(n/2)+1]  # Take only values on [0, pi]
    if window:
        I_w = smooth(I_w, window_len=window_len, window=window)
    return w, I_w


def ar_periodogram(x, window='hanning', window_len=7):
    """
    Compute periodogram from data x, using prewhitening, smoothing and
    recoloring.  The data is fitted to an AR(1) model for prewhitening,
    and the residuals are used to compute a first-pass periodogram with
    smoothing.  The fitted coefficients are then used for recoloring.

    Parameters
    ----------
    x : array_like(float)
        A flat NumPy array containing the data to smooth
    window_len : scalar(int), optional
        An odd integer giving the length of the window.  Defaults to 7.
    window : string
        A string giving the window type. Possible values are 'flat',
        'hanning', 'hamming', 'bartlett' or 'blackman'

    Returns
    -------
    w : array_like(float)
        Fourier frequencies at which periodogram is evaluated
    I_w : array_like(float)
        Values of periodogram at the Fourier frequencies

    """
    # === run regression === #
    x_lag = x[:-1]  # lagged x
    X = np.array([np.ones(len(x_lag)), x_lag]).T  # add constant

    y = np.array(x[1:])  # current x

    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)  # solve for beta hat
    e_hat = y - X @ beta_hat  # compute residuals
    phi = beta_hat[1]  # pull out phi parameter

    # === compute periodogram on residuals === #
    w, I_w = periodogram(e_hat, window=window, window_len=window_len)

    # === recolor and return === #
    I_w = I_w / np.abs(1 - phi * np.exp(1j * w))**2

    return w, I_w


@njit(cache=True)
def _compute_reflected_signal(x: np.ndarray, window_len: int) -> np.ndarray:
    # === Reflect x around x[0] and x[-1] prior to convolution === #
    k = window_len // 2
    xb = x[:k]   # First k elements
    xt = x[-k:]  # Last k elements
    s_len = len(x) + 2 * k
    s = np.empty(s_len, dtype=x.dtype)
    # Reflect xb
    for i in range(k):
        s[i] = xb[k - i - 1]
    # Assign x
    for i in range(len(x)):
        s[k + i] = x[i]
    # Reflect xt
    for i in range(k):
        s[k + len(x) + i] = xt[k - i - 1]
    return s

@njit(cache=True)
def _convolve_valid(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    N = len(s)
    M = len(w)
    out_len = N - M + 1
    result = np.empty(out_len, dtype=np.float64)
    wsum = 0.0
    for i in range(M):
        wsum += w[i]
    # Pre-divide window
    win = np.empty_like(w, dtype=np.float64)
    for i in range(M):
        win[i] = w[i] / wsum
    # Convolution, valid mode
    for i in range(out_len):
        acc = 0.0
        for j in range(M):
            acc += win[j] * s[i + j]
        result[i] = acc
    return result

@njit(cache=True)
def _hanning_window(window_len: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(window_len) / (window_len - 1))

@njit(cache=True)
def _hamming_window(window_len: int) -> np.ndarray:
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(window_len) / (window_len - 1))

@njit(cache=True)
def _bartlett_window(window_len: int) -> np.ndarray:
    n = window_len
    ret = np.empty(n, dtype=np.float64)
    mid = (n - 1) / 2.0
    for i in range(n):
        ret[i] = 1.0 - abs((i - mid) / mid)
    return ret

@njit(cache=True)
def _blackman_window(window_len: int) -> np.ndarray:
    n = window_len
    x = np.arange(n)
    return (0.42
            - 0.5 * np.cos(2.0 * np.pi * x / (n - 1))
            + 0.08 * np.cos(4.0 * np.pi * x / (n - 1)))

@njit(cache=True)
def _flat_window(window_len: int) -> np.ndarray:
    arr = np.empty(window_len, dtype=np.float64)
    for i in range(window_len):
        arr[i] = 1.0
    return arr
