from scipy import signal
import numpy as np


def _weighted_window_operation(data,
                               window_size,
                               window_function_values):
    if len(data) < window_size:
        data = np.concatenate((np.zeros(window_size-len(data)), data))
    return np.multiply(window_function_values, data)


def _barthann_window(data,
                     window_size,
                     symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.barthann(window_size, sym=symmetric))

 
def _weighted_moving_window(data,
                            window_size):
    return _weighted_window_operation(data, window_size, np.arange(1,window_size+1)/np.arange(1,window_size+1).sum() )


def _bartlett_window(data,
                     window_size,
                     symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.bartlett(window_size, sym=symmetric))


def _blackman_window(data,
                     window_size,
                     symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.blackman(window_size, sym=symmetric))


def _blackmanharris_window(data,
                           window_size,
                           symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.blackmanharris(window_size, sym=symmetric))


def _bohman_window(data,
                   window_size,
                   symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.bohman(window_size, sym=symmetric))


def _cosine_window(data,
                   window_size,
                   symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.cosine(window_size, sym=symmetric))


def _exponential_window(data,
                        window_size,
                        center,
                        tau,
                        symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.exponential(window_size, center=center,
                                      tau=tau, sym=symmetric))


def _flattop_window(data,
                    window_size,
                    symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.flattop(window_size, sym=symmetric))


def _gaussian_window(data,
                     window_size,
                     std,
                     symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.gaussian(window_size, std, sym=symmetric))


def _hamming_window(data,
                    window_size,
                    symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.hamming(window_size, sym=symmetric))


def _hann_window(data,
                 window_size,
                 symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.hann(window_size, sym=symmetric))


def _kaiser_window(data,
                   window_size,
                   beta,
                   symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.kaiser(window_size, beta, sym=symmetric))


def _parzen_window(data,
                   window_size,
                   symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.parzen(window_size, sym=symmetric))


def _triang_window(data,
                   window_size,
                   symmetric):
    return _weighted_window_operation(data, window_size, signal.windows.triang(window_size, sym=symmetric))


def _equal_window(data,
                  window_size):
    return _weighted_window_operation(data, window_size, np.ones(window_size))
