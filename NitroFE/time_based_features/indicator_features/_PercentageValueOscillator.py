import numpy as np
import pandas as pd
from typing import Union, Callable

from pandas.core.frame import DataFrame
from NitroFE.time_based_features.moving_average_features.moving_average_features import (
    ExponentialMovingFeature,
)


class PercentageValueOscillator:
    """ 
    Provided dataframe must be in ascending order.
    """
    def __init__(
        self,
        fast_period: int = 4,
        slow_period: int = 8,
        smoothing_period:int = 9,
        fast_operation: str = "mean",
        slow_operation: str = "mean",
        initialize_using_operation: bool = False,
        initialize_span: int = None,
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
    ):
        """
        Parameters
        ----------
        fast_period : int, optional
            specify decay in terms of span, for the fast moving feature , by default 4
        slow_period : int, optional
            specify decay in terms of span, for the slow moving feature, by default 8
        smoothing_period : int , optional
            specify decay in terms of span, for the smoothing ema, by default 9
        fast_operation : str, optional
            operation to be performed for the fast moving feature, by default 'mean'
        slow_operation : str, optional
            operation to be performed for the slow moving feature, by default 'mean'
        initialize_using_operation : bool, optional
            If True, then specified 'operation' is performed on the first 'initialize_span' values, and then the exponential moving average is calculated, by default False
        initialize_span : int, optional
            the span over which 'operation' would be performed for initialization, by default None
        ignore_na : bool, optional
            Ignore missing values when calculating weights, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None
        """
        self.span_fast = fast_period
        self.span_slow = slow_period
        self.span_smoothing=smoothing_period

        self.min_periods = min_periods

        self.ignore_na = ignore_na
        self.axis = axis
        self.times = times
        self.fast_operation = fast_operation
        self.slow_operation = slow_operation

        self.initialize_using_operation = initialize_using_operation
        self.initialize_span = initialize_span

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
    ):
        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Indicator features require past values for calculation.
            Use True, when calculating for training data  (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True

        """

        if first_fit:
            self._fast_em_object = ExponentialMovingFeature(
                span=self.span_fast,
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
                operation=self.fast_operation,
            )
            self._slow_em_object = ExponentialMovingFeature(
                span=self.span_slow,
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
                operation=self.slow_operation,
            )
            self._smoothing_object = ExponentialMovingFeature(
                span=self.span_smoothing,
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
                operation=self.slow_operation,
            )

        fast_em = self._fast_em_object.fit(dataframe=dataframe, first_fit=first_fit)
        slow_em = self._slow_em_object.fit(dataframe=dataframe, first_fit=first_fit)

        res = (slow_em - fast_em) / slow_em
        res = self._smoothing_object.fit(dataframe=res, first_fit=first_fit)
        return res
