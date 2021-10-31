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


class AverageTrueRange:
    def __init__(
        self,
        true_range_lookback: int = 4,
        average_true_range_span: int = 6,
        true_range_min_periods: int = None,
        average_true_range_periods: int = 1,
        return_true_range: bool = False,
    ):
        """
        Parameters
        ----------
        true_range_lookback : int, optional
            Size of the rolling window for true range value calculation, by default 4
        average_true_range_span : int, optional
            Size of the rolling window for average true range value calculation, by default 6
        true_range_min_periods : int, optional
            Minimum number of observations in window required to have a value for true range calculation, by default None
        average_true_range_periods : int, optional
            Minimum number of observations in window required to have a value for average true range calculation , by default 1
        return_true_range : bool, optional
            If true, True range is returned instead of Average True range
        """

        self.true_range_lookback = true_range_lookback
        self.average_true_range_span = average_true_range_span

        self.true_range_min_periods = true_range_min_periods
        self.average_true_range_periods = average_true_range_periods
        self.return_true_range = return_true_range

    def true_range(self, x):
        return np.max(
            [
                (np.max(x) - np.min(x)),
                np.abs(np.max(x) - x.iloc[-1]),
                np.abs(np.min(x) - x.iloc[-1]),
            ]
        )

    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], first_fit: bool = True):
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

            self._true_range_moving_average_object = weighted_window_features()
            self._average_true_range_moving_average_object = weighted_window_features()

        true_range = (
            self._true_range_moving_average_object._template_feature_calculation(
                function_name="true_range",
                win_function=_equal_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=self.true_range_lookback,
                min_periods=self.true_range_min_periods,
                symmetric=None,
                operation=self.true_range,
                operation_args=(),
            )
        )
        if self.return_true_range:
            return true_range

        average_true_range = self._average_true_range_moving_average_object._template_feature_calculation(
            function_name="average_true_range",
            win_function=_equal_window,
            first_fit=first_fit,
            dataframe=true_range,
            window=self.true_range_lookback,
            min_periods=self.average_true_range_periods,
            symmetric=None,
            operation=np.mean,
            operation_args=(),
        )

        return average_true_range
