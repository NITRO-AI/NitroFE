import numpy as np
import pandas as pd
from typing import Union, Callable

from pandas.core.frame import DataFrame
from NitroFE.time_based_features.indicator_features._AbsolutePriceOscillator import (
    AbsolutePriceOscillator,
)
from NitroFE.time_based_features.moving_average_features.moving_average_features import (
    ExponentialMovingFeature,
)


class MovingAverageConvergenceDivergence:
    """ 
    Provided dataframe must be in ascending order.
    """
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        smoothing_period: int = 9,
        fast_operation: str = "mean",
        slow_operation: str = "mean",
        smoothing_operation: str = "mean",
        initialize_using_operation: bool = False,
        initialize_span: int = None,
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
        return_histogram=False,
    ):
        """
        
        Parameters
        ----------
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
        initialize_using_operation : bool, optional
            If True, then specified 'operation' is performed on the first 'initialize_span' values, and then the exponential moving average is calculated, by default False
        initialize_span : int, optional
            the span over which 'operation' would be performed for initialization, by default None
        min_periods : int, optional
            Minimum number of observations in window required to have a value (otherwise result is NA), by default 0
        ignore_na : bool, optional
            Ignore missing values when calculating weights; specify True to reproduce pre-0.15.0 behavior, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None
        """

        self.span_fast = fast_period
        self.span_slow = slow_period
        self.min_periods = min_periods
        self.ignore_na = ignore_na
        self.axis = axis
        self.times = times
        self.fast_operation = fast_operation
        self.slow_operation = slow_operation
        self.smoothing_operation = smoothing_operation
        self.smoothing_period = smoothing_period
        self.return_histogram = return_histogram

        self.initialize_using_operation = initialize_using_operation
        self.initialize_span = initialize_span

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
    ):

        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

        Returns --> Smoothed signal line , macd histogram
        
        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Indicator features require past values for calculation.
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True



        """
        if first_fit:
            self._raw_macd_object = AbsolutePriceOscillator(
                fast_period=self.span_fast,
                slow_period=self.span_slow,
                fast_operation=self.fast_operation,
                slow_operation=self.slow_operation,
                min_periods=self.min_periods,
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
            )

            self._macd_object = ExponentialMovingFeature(
                span=self.smoothing_period,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
                operation=self.smoothing_operation,
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
            )

        raw_macd = self._raw_macd_object.fit(dataframe, first_fit=first_fit)

        macd = self._macd_object.fit(dataframe=raw_macd, first_fit=first_fit)

        return raw_macd - macd if self.return_histogram else macd
