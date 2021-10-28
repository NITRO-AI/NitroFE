
import numpy as np
import pandas as pd
from typing import Union, Callable
from NitroFE.time_based_features.weighted_window_features.weighted_window_features import (
    weighted_window_features,
)
from NitroFE.time_based_features.weighted_window_features.weighted_windows import (
    _equal_window,
    _identity_window,
)

class TypicalValue:
    def __init__(self,
            lookback_period: int = 6,
        min_periods: int = None):
        """
        Parameters
        ----------
        lookback_period : int, optional
            Size of the rolling window for lookback, by default 6
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default None
        """
        self.lookback_period = lookback_period
        self.min_periods = min_periods

    def _calculate_typical_value(self, x):
        return (np.max(x) + np.min(x) + x.iloc[-1:]) / 3

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True):
        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

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
            self._typical_value_object = weighted_window_features()


        _typical_value = self._typical_value_object._template_feature_calculation(
            function_name="typical_value",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._calculate_typical_value,
            operation_args=(),
        )

        return _typical_value