from scipy import signal
import numpy as np


def _weighted_window_operation(data,
                               window_size,
                               window_function_values):
    if len(data) < window_size:
        data = np.concatenate((np.zeros(window_size-len(data)), data))
    return np.multiply(window_function_values, data)


def _barthann_non_symmetric_window(data,
                                   window_size):
    return _weighted_window_operation(data, window_size, signal.windows.barthann(window_size, sym=False))


def _barthann_symmetric_window(data,
                               window_size):
    return _weighted_window_operation(data, window_size, signal.windows.barthann(window_size, sym=True))


def _bartlett_non_symmetric_window(data,
                                   window_size):
    return _weighted_window_operation(data, window_size, signal.windows.bartlett(window_size, sym=False))


def _bartlett_symmetric_window(data,
                               window_size):
    return _weighted_window_operation(data, window_size, signal.windows.bartlett(window_size, sym=True))


def _blackman_non_symmetric_window(data,
                                   window_size):
    return _weighted_window_operation(data, window_size, signal.windows.blackman(window_size, sym=False))


def _blackman_symmetric_window(data,
                               window_size):
    return _weighted_window_operation(data, window_size, signal.windows.blackman(window_size, sym=True))


def _blackmanharris_non_symmetric_window(data,
                                         window_size):
    return _weighted_window_operation(data, window_size, signal.windows.blackmanharris(window_size, sym=False))


def _blackmanharris_symmetric_window(data,
                                     window_size):
    return _weighted_window_operation(data, window_size, signal.windows.blackmanharris(window_size, sym=True))


def _bohman_non_symmetric_window(data,
                                 window_size):
    return _weighted_window_operation(data, window_size, signal.windows.bohman(window_size, sym=False))


def _bohman_symmetric_window(data,
                             window_size):
    return _weighted_window_operation(data, window_size, signal.windows.bohman(window_size, sym=True))


def _cosine_non_symmetric_window(data,
                                 window_size):
    return _weighted_window_operation(data, window_size, signal.windows.cosine(window_size, sym=False))


def _cosine_symmetric_window(data,
                             window_size):
    return _weighted_window_operation(data, window_size, signal.windows.cosine(window_size, sym=True))


def _exponential_non_symmetric_window(data,
                                      window_size,
                                      center,
                                      tau):
    return _weighted_window_operation(data, window_size, signal.windows.exponential(window_size, sym=False))


def _exponential_symmetric_window(data,
                                  window_size,
                                  center,
                                  tau):
    return _weighted_window_operation(data, window_size, signal.windows.exponential(window_size, sym=True))


def _flattop_non_symmetric_window(data,
                                  window_size):
    return _weighted_window_operation(data, window_size, signal.windows.flattop(window_size, sym=False))


def _flattop_symmetric_window(data,
                              window_size):
    return _weighted_window_operation(data, window_size, signal.windows.flattop(window_size, sym=True))


def _gaussian_non_symmetric_window(data,
                                   window_size,
                                   std):
    return _weighted_window_operation(data, window_size, signal.windows.gaussian(window_size, std, sym=False))


def _gaussian_symmetric_window(data,
                               window_size,
                               std):
    return _weighted_window_operation(data, window_size, signal.windows.gaussian(window_size, std, sym=True))


def _equal_window(data,
                  window_size):
    return _weighted_window_operation(data, window_size, np.ones(window_size))
