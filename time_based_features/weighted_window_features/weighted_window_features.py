from weighted_windows import _barthann_non_symmetric_window, _barthann_symmetric_window, _bartlett_non_symmetric_window,\
    _bartlett_symmetric_window, _equal_window, _blackman_symmetric_window, _blackman_non_symmetric_window, _blackmanharris_non_symmetric_window,\
    _blackmanharris_symmetric_window, _bohman_non_symmetric_window, _bohman_symmetric_window, _cosine_non_symmetric_window,\
    _cosine_symmetric_window, _exponential_symmetric_window, _exponential_non_symmetric_window, _flattop_non_symmetric_window,\
    _flattop_symmetric_window, _gaussian_symmetric_window, _gaussian_non_symmetric_window,_hamming_symmetric_window,_hamming_non_symmetric_window,\
        _hann_non_symmetric_window,_hann_symmetric_window,_kaiser_symmetric_window,_kaiser_non_symmetric_window,_parzen_non_symmetric_window,\
            _parzen_symmetric_window,_triang_symmetric_window,_triang_non_symmetric_window

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
        dataframe/series over which Bartlett–Hann weighted rolling window feature is to be constructed 
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

    References
    -----
    .. [1] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Bartlett%E2%80%93Hann_window

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
        dataframe/series over which bartlett weighted rolling window feature is to be constructed 
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

    References
    -----
    .. [1] Center for Computer Research in Music and Acoustics, "Bartlett (``Triangular'') Window",
           https://ccrma.stanford.edu/~jos/sasp/Bartlett_Triangular_Window.html
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
        dataframe/series over which equally weighted rolling window feature is to be constructed 
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
        dataframe/series over which blackman weighted rolling window feature is to be constructed 
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

    References
    -----
    .. [1] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Blackman_window
    """
    win_function = _blackman_symmetric_window if symmetric else _blackman_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))


def caluclate_blackmanharris_feature(dataframe: Union[pd.DataFrame, pd.Series],
                                     window: int = 3,
                                     min_periods: int = 1,
                                     symmetric: bool = False,
                                     operation: Callable = np.mean):
    """
    Create blackman-harris weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which blackman-harris weighted rolling window feature is to be constructed 
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

    References
    -----
    .. [1] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window 
    """
    win_function = _blackmanharris_symmetric_window if symmetric else _blackmanharris_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))


def caluclate_bohman_feature(dataframe: Union[pd.DataFrame, pd.Series],
                             window: int = 3,
                             min_periods: int = 1,
                             symmetric: bool = False,
                             operation: Callable = np.mean):
    """
    Create bohman weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which bohman weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] mathworks, "bohmanwin",
           https://www.mathworks.com/help/signal/ref/bohmanwin.html 

    """
    win_function = _bohman_symmetric_window if symmetric else _bohman_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))


def caluclate_cosine_feature(dataframe: Union[pd.DataFrame, pd.Series],
                             window: int = 3,
                             min_periods: int = 1,
                             symmetric: bool = False,
                             operation: Callable = np.mean):
    """
    Create cosine weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which cosine weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Power-of-sine/cosine_windows

    """
    win_function = _cosine_symmetric_window if symmetric else _cosine_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))


def caluclate_exponential_feature(dataframe: Union[pd.DataFrame, pd.Series],
                                  window: int = 3,
                                  min_periods: int = 1,
                                  symmetric: bool = False,
                                  center: float = None,
                                  tau: float = 1,
                                  operation: Callable = np.mean):
    """
    Create exponential weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which exponential weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    center : float , optional
        Parameter defining the center location of the window function.
         The default value if not given is center = (M-1) / 2. This parameter must take its default value for symmetric windows.
    tau : float , optional
        Parameter defining the decay. For center = 0 use tau = -(M-1) / ln(x) if x is the fraction of the window remaining at the end, by default 1
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Exponential_or_Poisson_window

    """
    win_function = _exponential_symmetric_window if symmetric else _exponential_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window, center, tau)))


def caluclate_flattop_feature(dataframe: Union[pd.DataFrame, pd.Series],
                              window: int = 3,
                              min_periods: int = 1,
                              symmetric: bool = False,
                              operation: Callable = np.mean):
    """
    Create flattop weighted rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which flattop weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] mathworks, "flattopwin",
           https://in.mathworks.com/help/signal/ref/flattopwin.html

    """
    win_function = _flattop_symmetric_window if symmetric else _flattop_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))


def caluclate_gaussian_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               std: float = 1,
                               operation: Callable = np.mean):
    """
    Create flattop gaussian rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which flattop gaussian rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    std : float, optional
        The standard deviation, sigma.
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Gaussian_window

    """
    win_function = _gaussian_symmetric_window if symmetric else _gaussian_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window, std)))


def caluclate_hamming_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               operation: Callable = np.mean):
    """
    Create flattop hamming rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which flattop hamming rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Gaussian_window

    """
    win_function = _hamming_symmetric_window if symmetric else _hamming_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))

def caluclate_hann_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               operation: Callable = np.mean):
    """
    Create flattop hann rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which flattop hann rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] Wikipedia, "Hann function",
           https://en.wikipedia.org/wiki/Hann_function

    """
    win_function = _hann_symmetric_window if symmetric else _hann_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))

def caluclate_kaiser_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               beta : float=7,
                               operation: Callable = np.mean):
    """
    Create flattop kaiser rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which flattop kaiser rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    beta : float, optional
        Shape parameter, determines trade-off between main-lobe width and side lobe level, by default 7
         As beta gets large, the window narrows.
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] Wikipedia, "Kaiser window",
           https://en.wikipedia.org/wiki/Kaiser_window
    """
    win_function = _kaiser_symmetric_window if symmetric else _kaiser_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window,beta)))   

def caluclate_parzen_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               operation: Callable = np.mean):
    """
    Create flattop parzen rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which flattop parzen rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] Wikipedia, "Parzen function",
           https://en.wikipedia.org/wiki/Window_function#Parzen_window

    """
    win_function = _parzen_symmetric_window if symmetric else _parzen_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))

def caluclate_triang_feature(dataframe: Union[pd.DataFrame, pd.Series],
                               window: int = 3,
                               min_periods: int = 1,
                               symmetric: bool = False,
                               operation: Callable = np.mean):
    """
    Create flattop triang rolling window feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over which flattop triang rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    symmetric : bool, optional
        When True , generates a symmetric window, for use in filter design. When False,
        generates a periodic window, for use in spectral analysis, by default False
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    References
    -----
    .. [1] Wikipedia, "triang function",
           https://en.wikipedia.org/wiki/Window_function#Parzen_window

    """
    win_function = _triang_symmetric_window if symmetric else _triang_non_symmetric_window
    return dataframe.rolling(window, min_periods=min_periods).agg(lambda x: operation(win_function(x, window)))