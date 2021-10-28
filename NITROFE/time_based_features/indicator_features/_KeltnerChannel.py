from NitroFE.time_based_features.indicator_features._AverageTrueRange import (
    AverageTrueRange,
)
import numpy as np
import pandas as pd
from typing import Union
from NitroFE.time_based_features.weighted_window_features.weighted_window_features import (
    weighted_window_features,
)
from NitroFE.time_based_features.weighted_window_features.weighted_windows import (
    _equal_window,
    _identity_window,
)
from NitroFE.time_based_features.moving_average_features.moving_average_features import (
    ExponentialMovingFeature,
)

class KeltnerChannel:
    def __init__(
        self,
        ema_span: int = 8,
        initialize_using_operation: bool = False,
        initialize_span: int = None,
        true_range_lookback: int = 4,
        average_true_range_span: int = 6,
        true_range_min_periods: int = None,
        average_true_range_periods: int = 1,
        atr_multiply: int = 2,
    ):
        """
        Parameters
        ----------
        ema_span : int, optional
            specify decay in terms of span for exponential moving average, by default 8
        initialize_using_operation : bool, optional
            If True, then specified mean is performed over the first 'initialize_span' values, and then the exponential moving average is calculated, by default False
        initialize_span : int, optional
            the span over which 'operation' would be performed for initialization, by default None
        true_range_lookback : int, optional
            Size of the rolling window for true range value calculation, by default 4
        average_true_range_span : int, optional
            Size of the rolling window for average true range value calculation, by default 6
        true_range_min_periods : int, optional
            Minimum number of observations in window required to have a value for true range calculation, by default None
        average_true_range_periods : int, optional
            Minimum number of observations in window required to have a value for average true range calculation , by default 1
        atr_multiply : int, optional
            multiplication factor for average true range, by default 2
        """
        self.ema_span = ema_span
        self.initialize_using_operation = initialize_using_operation
        self.initialize_span = initialize_span
        self.true_range_lookback = true_range_lookback
        self.average_true_range_span = average_true_range_span
        self.true_range_min_periods = true_range_min_periods
        self.average_true_range_periods = average_true_range_periods
        self.atr_multiply = atr_multiply

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
            self._atr_object = AverageTrueRange(
                true_range_lookback=self.true_range_lookback,
                average_true_range_span=self.average_true_range_span,
                true_range_min_periods=self.true_range_min_periods,
                average_true_range_periods=self.average_true_range_periods,
            )
            self._ema_object = ExponentialMovingFeature(
                span=self.ema_span,
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
            )

        atr_val = self._atr_object.fit(dataframe=dataframe, first_fit=first_fit)

        ema_val = self._ema_object.fit(dataframe=dataframe, first_fit=first_fit)

        positive_band = ema_val + self.atr_multiply * atr_val
        negative_band = ema_val - self.atr_multiply * atr_val

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
