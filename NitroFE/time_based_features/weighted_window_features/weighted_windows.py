from scipy import signal
import numpy as np


def _weighted_window_operation(data,
                               window_size,
                               window_function_values,
                               resize=True):
    
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
    else:
        window_function_values=window_function_values[:len(data)]
    return np.multiply(window_function_values, data)


def _barthann_window(data,
                     window_size,
                     symmetric,
                    resize=False):

    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.barthann(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.barthann(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)   

 
def _weighted_moving_window(data,
                            window_size,
                            symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=np.arange(1,window_size+1)/np.arange(1,window_size+1).sum()
    else:
        window_function_values=np.arange(1,len(data)+1)/np.arange(1,len(data)+1).sum()
    return np.multiply(window_function_values, data)  


def _bartlett_window(data,
                     window_size,
                     symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.bartlett(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.bartlett(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)   


def _blackman_window(data,
                     window_size,
                     symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.blackman(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.blackman(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)  


def _blackmanharris_window(data,
                           window_size,
                           symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.blackmanharris(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.blackmanharris(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)


def _bohman_window(data,
                   window_size,
                   symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.bohman(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.bohman(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)


def _cosine_window(data,
                   window_size,
                   symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.cosine(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.cosine(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)


def _exponential_window(data,
                        window_size,
                        center,
                        tau,
                        symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.exponential(window_size, center=center,
                                      tau=tau, sym=symmetric)
    else:
        window_function_values=signal.windows.exponential(len(data), center=center,
                                      tau=tau, sym=symmetric)
    return np.multiply(window_function_values, data)



def _flattop_window(data,
                    window_size,
                    symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.flattop(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.flattop(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)


def _gaussian_window(data,
                     window_size,
                     std,
                     symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.gaussian(window_size, std=std,sym=symmetric)
    else:
        window_function_values=signal.windows.gaussian(len(data), std=std,sym=symmetric)
    return np.multiply(window_function_values, data)


def _hamming_window(data,
                    window_size,
                    symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.hamming(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.hamming(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)


def _hann_window(data,
                 window_size,
                 symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.hamming(window_size, sym=symmetric)
    else:
        window_function_values=signal.windows.hamming(len(data), sym=symmetric)
    return np.multiply(window_function_values, data)


def _kaiser_window(data,
                   window_size,
                   beta,
                   symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.kaiser(window_size, beta,sym=symmetric)
    else:
        window_function_values=signal.windows.kaiser(len(data), beta,sym=symmetric)
    return np.multiply(window_function_values, data)


def _parzen_window(data,
                   window_size,
                   symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.parzen(window_size,sym=symmetric)
    else:
        window_function_values=signal.windows.parzen(len(data),sym=symmetric)
    return np.multiply(window_function_values, data)


def _triang_window(data,
                   window_size,
                   symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=signal.windows.triang(window_size,sym=symmetric)
    else:
        window_function_values=signal.windows.triang(len(data),sym=symmetric)
    return np.multiply(window_function_values, data)


def _equal_window(data,
                  window_size,
                  symmetric,
                  resize=False):
    if (len(data) < window_size)&(resize):
        data = np.concatenate((np.zeros(window_size-len(data)), data))
        window_function_values=np.ones(window_size)
    else:
        window_function_values=np.ones(len(data))
    return np.multiply(window_function_values, data)

def _identity_window(data,
                  window_size,
                  symmetric,
                  resize=False):
    return data
