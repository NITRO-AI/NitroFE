from weighted_windows import _barthann_non_symmetric_window, _barthann_symmetric_window, _bartlett_non_symmetric_window,\
    _bartlett_symmetric_window, _equal_window, _blackman_symmetric_window, _blackman_non_symmetric_window
import numpy as np
import pandas as pd
from typing import Union, Callable


def caluclate_barthann_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               operation: Callable = np.mean):
    """
    Create Bartlett–Hann weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which barthann weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    Returns
    -------
        final value after operation

    Notes
    -----
    Make sure the dataframe/Series is sorted. 
    The Bartlett window is defined as
    .. math:: w(n) = \\frac{2}{M-1} \\left(
              \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|
              \\right)
    Most references to the Bartlett window come from the signal
    processing literature, where it is used as one of many windowing
    functions for smoothing values.  Note that convolution with this
    window produces linear interpolation.  It is also known as an
    apodization (which means"removing the foot", i.e. smoothing
    discontinuities at the beginning and end of the sampled signal) or
    tapering function. The fourier transform of the Bartlett is the product
    of two sinc functions.

    """
    win_function = _barthann_symmetric_window if symmetric else _barthann_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))


def caluclate_bartlett_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               operation: Callable = np.mean):
    """
    Create bartlett weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which barthann weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    Returns
    -------
        final value after operation

    Notes
    -----
    The Bartlett window is defined as
    .. math:: w(n) = \frac{2}{M-1} \left(
              \frac{M-1}{2} - \left|n - \frac{M-1}{2}\right|
              \right)
    Most references to the Bartlett window come from the signal
    processing literature, where it is used as one of many windowing
    functions for smoothing values.  Note that convolution with this
    window produces linear interpolation.  It is also known as an
    apodization (which means"removing the foot", i.e. smoothing
    discontinuities at the beginning and end of the sampled signal) or
    tapering function. The Fourier transform of the Bartlett is the product
    of two sinc functions.

    """
    win_function = _bartlett_symmetric_window if symmetric else _bartlett_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))


def caluclate_equal_feature(dataframe: Union[pd.DataFrame, pd.Series],
                            window: int = 3,
                            min_periods: int = 1,
                            symmetric: bool = False,
                            operation: Callable = np.mean):
    """
    Create equally weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which barthann weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    Returns
    -------
        final value after operation

    Notes
    -----
    all elements in the window are weighted equally

    """
    win_function = _equal_window if symmetric else _equal_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))


def caluclate_blackman_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               operation: Callable = np.mean):
    """
    Create blackman weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which barthann weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    Returns
    -------
        final value after operation

    Notes
    -----
     The Blackman window is defined as
    .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)
    The "exact Blackman" window was designed to null out the third and fourth
    sidelobes, but has discontinuities at the boundaries, resulting in a
    6 dB/oct fall-off.  This window is an approximation of the "exact" window,
    which does not null the sidelobes as well, but is smooth at the edges,
    improving the fall-off rate to 18 dB/oct. [3]_
    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the Kaiser window.
    """
    win_function = _blackman_symmetric_window if symmetric else _blackman_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))
