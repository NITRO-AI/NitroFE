import pandas as pd
import numpy as np
from typing import Union, Callable

from NitroFE.time_based_features.weighted_window_features.weighted_window_features import caluclate_weighted_moving_window_feature, caluclate_equal_feature


def calculate_exponential_moving_feature(dataframe: Union[pd.DataFrame, pd.Series],
                                         com: float = None,
                                         span: float = None,
                                         halflife: float = None,
                                         alpha: float = None,
                                         min_periods: int = 0,
                                         adjust: bool = False,
                                         ignore_na: bool = False,
                                         axis: int = 0,
                                         times: str = None,
                                         operation: str = 'mean'):
    """
    simple wrapper for pandas.ewm function
    kindly refer to https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    """
    _dataframe = dataframe.ewm(com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods,
                               adjust=adjust, ignore_na=ignore_na, axis=axis, times=times)
    _return = _dataframe.mean() if operation == 'mean' else _dataframe.var(
    ) if operation == 'var' else _dataframe.std() if operation == 'std' else None
    if _return is None:
        raise ValueError(f"Operation {operation} not supported")
    return _return


def calculate_weighted_moving_feature(dataframe: Union[pd.DataFrame, pd.Series],
                                      window: int = 3,
                                      min_periods: int = 1,
                                      operation: Callable = np.sum):
    """
    Create weighted moving average feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    Returns
    -------
        final value after operation

    References
    -----
    .. [1] Wikipedia, "Weighted Window function",
           https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

    """

    return caluclate_weighted_moving_window_feature(dataframe=dataframe, window=window, min_periods=min_periods, operation=operation)


def calculate_simple_moving_feature(dataframe: Union[pd.DataFrame, pd.Series],
                                    window: int = 3,
                                    min_periods: int = 1,
                                    operation: Callable = np.mean):
    """
    Create simple moving average feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    Returns
    -------
        final value after operation

    References
    -----
    .. [1] Wikipedia, "Weighted Window function",
           https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

    """

    return caluclate_equal_feature(dataframe, window, min_periods, operation)


def calculate_hull_moving_feature(dataframe: Union[pd.DataFrame, pd.Series],
                                  window: int = 4,
                                  min_periods: int = 1,
                                  operation: Callable = np.sum):
    """
    Create hull moving average feature

    Parameters
    ----------
    dataframe : Union[pd.DataFrame,pd.Series]
        dataframe/series over weighted rolling window feature is to be constructed 
    window : int, optional
        Size of the rolling window, by default 3
    min_periods : int, optional
        Minimum number of observations in window required to have a value, by default 1
    operation : Callable, optional
        operation to perform over the weighted rolling window values, by default np.mean

    Returns
    -------
        final value after operation

    References
    -----
    .. [1] Wikipedia, "Weighted Window function",
           https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

    """

    if window <= 1:
        raise ValueError(f'window size less than equal to 1 not supported')
    window_by_two, window_square_root = int(
        np.ceil(window/2)), int(np.ceil(np.sqrt(window)))

    window_size_weighted_moving_average = calculate_weighted_moving_feature(
        dataframe=dataframe, window=window, min_periods=min_periods, operation=operation)

    window_by_two_size_weighted_moving_average = calculate_weighted_moving_feature(
        dataframe=dataframe, window=window_by_two, min_periods=min_periods, operation=operation)

    raw_hma = 2*window_by_two_size_weighted_moving_average - \
        window_size_weighted_moving_average
    hma = calculate_weighted_moving_feature(
        dataframe=raw_hma, window=window_square_root, min_periods=min_periods, operation=operation)

    return hma


def calculate_triple_exponential_moving_feature(dataframe: Union[pd.DataFrame, pd.Series],
                                                com: float = None,
                                                span: float = None,
                                                halflife: float = None,
                                                alpha: float = None,
                                                min_periods: int = 0,
                                                adjust: bool = True,
                                                ignore_na: bool = False,
                                                axis: int = 0,
                                                times: str = None,
                                                operation: str = 'mean'):
    """
    Create triple moving average feature

    """
    first_exponential_average = calculate_exponential_moving_feature(dataframe=dataframe, com=com, span=span, halflife=halflife,
                                                                     alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis,
                                                                     times=times, operation=operation)
    second_exponential_average = calculate_exponential_moving_feature(dataframe=first_exponential_average, com=com, span=span, halflife=halflife,
                                                                     alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis,
                                                                     times=times, operation=operation)
    third_exponential_average = calculate_exponential_moving_feature(dataframe=second_exponential_average, com=com, span=span, halflife=halflife,
                                                                     alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis,
                                                                     times=times, operation=operation)
    triple_exponential_average=3*first_exponential_average-3*second_exponential_average+third_exponential_average
    return triple_exponential_average



print(calculate_triple_exponential_moving_feature(
    pd.DataFrame({'a': np.arange(10), 'b': 2+np.arange(10)}), span= 3,operation='mean'))
