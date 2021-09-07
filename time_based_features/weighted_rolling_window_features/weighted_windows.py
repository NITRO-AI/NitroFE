from scipy import signal
import numpy as np
import pandas as pd
from typing import Union, Callable


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

def _equal_window(data,
                  window_size):
    return _weighted_window_operation(data, window_size, np.ones(window_size))
