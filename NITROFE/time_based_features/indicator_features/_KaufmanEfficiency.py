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


class KaufmanEfficiency:
    def __init__(self, lookback_period: int = 4, min_periods: int = None):
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

    def _calculate_kaufman_efficiency(self, x):
        up = np.abs(x.iloc[-1] - x.iloc[0])
        down = np.abs(x.diff().fillna(0)).sum()
        return up / down

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
            self._kaufman_efficiency_object = weighted_window_features()

        _kaufman_efficiency = (
            self._kaufman_efficiency_object._template_feature_calculation(
                function_name="kaufman_efficiency",
                win_function=_identity_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=self.lookback_period,
                min_periods=self.min_periods,
                symmetric=None,
                operation=self._calculate_kaufman_efficiency,
                operation_args=(),
            )
        )

        return _kaufman_efficiency
