import numpy as np
import pandas as pd
from typing import Union, Callable
from NitroFE.time_based_features.moving_average_features.moving_average_features import calculate_exponential_moving_feature


def calculate_absolute_price_oscillator(dataframe: Union[pd.DataFrame, pd.Series],
                                        fast_period: int = 10,
                                        slow_period: int = 20,
                                        fast_operation: str = 'mean',
                                        slow_operation: str = 'mean',
                                        adjust: bool = False,
                                        ignore_na: bool = False,
                                        axis: int = 0,
                                        times: str = None):
    """calculate_absolute_price_oscillator 

    The Absolute Price Oscillator displays the difference between two exponential moving averages,
     of a security's price and is expressed as an absolute value.

    Parameters
    ----------
    dataframe : Union[pd.DataFrame, pd.Series] 
        dataframe containing column values to create feature over
    fast_period : int, optional
        specify decay in terms of span, for the fast moving feature , by default 10
    slow_period : int, optional
        specify decay in terms of span, for the slow moving feature, by default 20
    fast_operation : str, optional
        operation to be performed for the fast moving feature, by default 'mean'
    slow_operation : str, optional
        operation to be performed for the slow moving feature, by default 'mean'
    adjust : bool, optional
        Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings (viewing EWMA as a moving average)
        , When adjust=False, the exponentially weighted function is calculated recursively,by default False
    ignore_na : bool, optional
        gnore missing values when calculating weights, by default False
    axis : int, optional
        The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
    times : str, optional
        Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None

    Returns
    -------
    absolute price oscillator

    References
    -----
    .. [1] fidelity, "apo",
           https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo
    """
    fast_em = calculate_exponential_moving_feature(
        dataframe=dataframe, span=fast_period, adjust=adjust, ignore_na=ignore_na, axis=axis, times=times, operation=fast_operation)
    slow_em = calculate_exponential_moving_feature(
        dataframe=dataframe, span=slow_period, adjust=adjust, ignore_na=ignore_na, axis=axis, times=times, operation=slow_operation)

    absolute_price_oscillator = fast_em-slow_em
    return absolute_price_oscillator


def calculate_moving_average_convergence_divergence(dataframe: Union[pd.DataFrame, pd.Series],
                                                    fast_period: int = 12,
                                                    slow_period: int = 26,
                                                    smoothing_period: int = 9,
                                                    fast_operation: str = 'mean',
                                                    slow_operation: str = 'mean',
                                                    smoothing_operation: str = 'mean',
                                                    adjust: bool = False,
                                                    ignore_na: bool = False,
                                                    axis: int = 0,
                                                    times: str = None):
    """calculate_moving_average_convergence_divergence [summary]

    Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship 
    between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential 
    moving average (EMA) from the 12-period EMA.

    The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line," 
    is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals. 

    Parameters
    ----------
    dataframe : Union[pd.DataFrame, pd.Series]
        dataframe containing column values to create feature over
    fast_period : int, optional
        specify decay in terms of span, for the fast moving feature, by default 12
    slow_period : int, optional
        specify decay in terms of span, for the slow moving feature, by default 26
    smoothing_period : int, optional
        specify decay in terms of span, for the smoothing moving feature, by default 9
    fast_operation : str, {'mean','var','std'}
        operation to be performed for the fast moving feature, by default 'mean'
    slow_operation : str, {'mean','var','std'}
        operation to be performed for the slow moving feature, by default 'mean'
    smoothing_operation : str, optional
        operation to be performed for the smoothing moving feature, by default 'mean'
    adjust : bool, optional
        Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings (viewing EWMA as a moving average)
        , When adjust=False, the exponentially weighted function is calculated recursively,by default False
    ignore_na : bool, optional
        gnore missing values when calculating weights, by default False
    axis : int, optional
        The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
    times : str, optional
        Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None

    Returns
    -------
    Smoothed signal line , macd histogram

    References
    -----
    .. [1] investopedia, "Moving Average Convergence Divergence (MACD)",
           https://www.investopedia.com/terms/m/macd.asp#:~:text=The%20MACD%20is%20calculated%20by,for%20buy%20and%20sell%20signals.
    """

    raw_macd = calculate_absolute_price_oscillator(dataframe, fast_period, slow_period, fast_operation, slow_operation,
                                                   adjust=adjust, ignore_na=ignore_na, axis=axis, times=times)

    macd = calculate_exponential_moving_feature(dataframe=raw_macd, span=smoothing_period,
                                                adjust=adjust, ignore_na=ignore_na, axis=axis, times=times, operation=smoothing_operation)

    return macd, raw_macd-macd

