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
from NitroFE.time_based_features.indicator_features._RelativeStrengthIndex import (
    RelativeStrengthIndex,
)


class InverseFisherRelativeStrengthIndex:
    def __init__(self, lookback_period: int = 8, lookback_for_inverse_fisher: int = 8):
        """
        Parameters
        ----------
        lookback_period : int, optional
            Size of the rolling window for lookback, by default 8
        lookback_for_inverse_fisher : int, optional
            Size of the rolling window for lookback for weighted moving average of rsi, by default 8
        """
        self.lookback_period = lookback_period
        self.lookback_for_inverse_fisher = lookback_for_inverse_fisher

    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], first_fit: bool = True):
        """
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
            self._rsi_object = RelativeStrengthIndex(
                lookback_period=self.lookback_period
            )
            self._ww_object = weighted_window_features()

        rsi_value = self._rsi_object.fit(dataframe, first_fit=first_fit)

        rsi_value = 0.1 * (rsi_value - 50)

        rsi_value = self._ww_object.caluclate_weighted_moving_window_feature(
            dataframe=rsi_value,
            first_fit=first_fit,
            window=self.lookback_for_inverse_fisher,
            operation=np.sum,
        )

        rsi_value = (np.exp(2 * rsi_value) - 1) / (np.exp(2 * rsi_value) + 1)
        return rsi_value
