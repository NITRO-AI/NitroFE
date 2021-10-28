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
from NitroFE.time_based_features.indicator_features._TypicalValue import TypicalValue


class BollingerBands:
    def __init__(
        self,
        typical_value_lookback_period: int = 6,
        typical_value_min_periods: int = None,
        moving_average_typical_value_lookback_period: int = 6,
        moving_average_typical_value_min_periods: int = None,
        standard_deviation_multiplier: int = 2,
    ):
        """
        Parameters
        ----------
        typical_value_lookback_period : int, optional
            Size of the rolling window for typical value, by default 6
        typical_value_min_periods : int, optional
            Minimum number of observations in window required to have a value for typical value, by default None
        moving_average_typical_value_lookback_period : int, optional
            Size of the rolling window for moving average of typical value, by default 6
        moving_average_typical_value_min_periods : int, optional
            Minimum number of observations in window required to have a value for moving average of typical value, by default None
        standard_deviation_multiplier : int, optional
            standard deviation multiplier for upper and lower bollinger band, by default 2
        """
        self.typical_value_lookback_period = typical_value_lookback_period
        self.typical_value_min_periods = typical_value_min_periods

        self.moving_average_typical_value_lookback_period = (
            moving_average_typical_value_lookback_period
        )
        self.moving_average_typical_value_min_periods = (
            moving_average_typical_value_min_periods
        )

        self.standard_deviation_multiplier = standard_deviation_multiplier

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
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True

        """

        if first_fit:
            self._typical_value_object = TypicalValue(
                lookback_period=self.typical_value_lookback_period,
                min_periods=self.typical_value_min_periods,
            )
            self._ma_bollinger_bands_object = weighted_window_features()
            self._std_bollinger_bands_object = weighted_window_features()

        _typical_value = self._typical_value_object.fit(
            dataframe=dataframe, first_fit=first_fit
        )

        _moving_average_typical_value = (
            self._ma_bollinger_bands_object._template_feature_calculation(
                function_name="moving_average_typical_value",
                win_function=_identity_window,
                first_fit=first_fit,
                dataframe=_typical_value,
                window=self.moving_average_typical_value_lookback_period,
                min_periods=self.moving_average_typical_value_min_periods,
                symmetric=None,
                operation=np.mean,
                operation_args=(),
            )
        )

        _std_typical_value = (
            self._std_bollinger_bands_object._template_feature_calculation(
                function_name="moving_average_typical_value",
                win_function=_identity_window,
                first_fit=first_fit,
                dataframe=_typical_value,
                window=self.moving_average_typical_value_lookback_period,
                min_periods=self.moving_average_typical_value_min_periods,
                symmetric=None,
                operation=np.std,
                operation_args=(),
            )
        )

        positive_band = (
            _moving_average_typical_value
            + self.standard_deviation_multiplier * _std_typical_value
        )

        negative_band = (
            _moving_average_typical_value
            - self.standard_deviation_multiplier * _std_typical_value
        )

        if isinstance(dataframe, pd.Series):
            if dataframe.name is None:
                positive_band.name = "positive_band"
                negative_band.name = "negative_band"
            else:
                positive_band.name = positive_band.name + "_positive_band"
                negative_band.name = negative_band.name + "_negative_band"
        elif isinstance(dataframe, pd.DataFrame):
            positive_band.columns = positive_band.columns + "_positive_band"
            negative_band.columns = negative_band.columns + "_negative_band"
        return pd.concat([positive_band, negative_band], axis=1)
