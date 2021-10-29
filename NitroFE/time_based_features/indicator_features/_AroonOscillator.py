import numpy as np
import pandas as pd
from typing import Union, Callable
from NitroFE.time_based_features.weighted_window_features.weighted_windows import (
    _equal_window,
    _identity_window,
)

from NitroFE.time_based_features.weighted_window_features.weighted_window_features import (
    weighted_window_features,
)


class AroonOscillator:
    def __init__(
        self,
        lookback_period: int = 4,
        min_periods: int = None,
    ):
        """
        Parameters
        ----------
        lookback_period : int, optional
            Size of the rolling window for lookback, by default 4
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default None
        """
        self.lookback_period = lookback_period
        self.min_periods = min_periods

    def _calculate_aroon_up(self, x, look_back_period):
        return x.argmax() / (look_back_period)

    def _calculate_aroon_down(self, x, look_back_period):
        return x.argmin() / (look_back_period)

    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], first_fit: bool = True):
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
            self._aroon_up_object = weighted_window_features()
            self._aroon_down_object = weighted_window_features()

        aroon_up = self._aroon_up_object._template_feature_calculation(
            function_name="aroon_up",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._calculate_aroon_up,
            operation_args=(self.lookback_period,),
        )
        aroon_down = self._aroon_up_object._template_feature_calculation(
            function_name="aroon_down",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._calculate_aroon_down,
            operation_args=(self.lookback_period,),
        )
        aroon_value = 100 * (aroon_up - aroon_down)

        return aroon_value
