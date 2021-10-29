import numpy as np
import pandas as pd
from typing import Union, Callable

from pandas.core.frame import DataFrame
from NitroFE.time_based_features.indicator_features._AverageTrueRange import (
    AverageTrueRange,
)
from NitroFE.time_based_features.moving_average_features.moving_average_features import (
    ExponentialMovingFeature,
)
from NitroFE.time_based_features.weighted_window_features.weighted_windows import (
    _equal_window,
    _identity_window,
)
from NitroFE.time_based_features.weighted_window_features.weighted_window_features import (
    weighted_window_features,
)


class AverageDirectionalMovementIndex:
    def __init__(self,
            directional_movement_lookback_period: int = 4,
        directional_movement_min_periods: int = None,
        directional_movement_smoothing_period: int = 14,
        directional_movement_smoothing_min_periods: int = None,
        average_directional_movement_smoothing_period: int = 14,
        average_directional_movement_min_periods: int = None,
        true_range_lookback: int = 4,
        average_true_range_span: int = 6,
        true_range_min_periods: int = None,
        average_true_range_periods: int = 1):
        """
        Parameters
        ----------
        directional_movement_lookback_period : int, optional
            window size for directional movement lookback, used for calculating +DM and -DM, by default 4
        directional_movement_min_periods : int, optional
            min periods size for directional movement lookback, used for calculating +DM and -DM, by default None
        directional_movement_smoothing_period : int, optional
            smoothing window period to be used for +DM and -DM smoothing, by default 14
        directional_movement_smoothing_min_periods : int, optional
            min periods size to be used for +DM and -DM smoothing, by default None
        average_directional_movement_smoothing_period : int, optional
            smoothing window period to be used for raw ADX, by default 14
        average_directional_movement_min_periods : int, optional
             min periods size to be used for raw ADX, by default None
        true_range_lookback : int, optional
            Size of the rolling window for true range value calculation, by default 4
        average_true_range_span : int, optional
            Size of the rolling window for average true range value calculation, by default 6
        true_range_min_periods : int, optional
            Minimum number of observations in window required to have a value for true range calculation, by default None
        average_true_range_periods : int, optional
            Minimum number of observations in window required to have a value for average true range calculation , by default 1
        """
        self.directional_movement_lookback_period = (
            directional_movement_lookback_period
        )
        self.directional_movement_min_periods = directional_movement_min_periods
        self.directional_movement_smoothing_period = (
            directional_movement_smoothing_period
        )
        self.directional_movement_smoothing_min_periods = (
            directional_movement_smoothing_min_periods
        )

        self.average_directional_movement_smoothing_period = (
            average_directional_movement_smoothing_period
        )
        self.average_directional_movement_min_periods = (
            average_directional_movement_min_periods
        )

        self.true_range_lookback = true_range_lookback
        self.average_true_range_span = average_true_range_span
        self.true_range_min_periods = true_range_min_periods
        self.average_true_range_periods = average_true_range_periods

    def _plus_dm(self, x, look_back_period):
        look_back_period=int(look_back_period/2)
        return np.max(x.iloc[look_back_period:]) - np.max(
            x.iloc[0 : look_back_period - 1]
        )

    def _minus_dm(self, x, look_back_period):
        look_back_period=int(look_back_period/2)
        return np.min(x.iloc[0 : look_back_period - 1]) - np.min(
            x.iloc[look_back_period:]
        )

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

            self._plus_dma_object = weighted_window_features()
            self._minus_dma_object = weighted_window_features()

            self._plus_dm_smoothing_object = ExponentialMovingFeature(
                alpha=1 / (self.directional_movement_smoothing_period),
                min_periods=self.directional_movement_smoothing_min_periods,
                operation="mean",
            )
            self._minus_dm_smoothing_object = ExponentialMovingFeature(
                alpha=1 / (self.directional_movement_smoothing_period),
                min_periods=self.directional_movement_smoothing_min_periods,
                operation="mean",
            )

            self._average_true_range_object = AverageTrueRange(
                true_range_lookback=self.true_range_lookback,
                average_true_range_span=self.average_true_range_span,
                true_range_min_periods=self.true_range_min_periods,
                average_true_range_periods=self.average_true_range_periods,
            )

            self._average_dm_smoothing_object = ExponentialMovingFeature(
                alpha=1 / (self.average_directional_movement_smoothing_period),
                min_periods=self.average_directional_movement_min_periods,
                operation="mean",
            )



        plus_dma = self._plus_dma_object._template_feature_calculation(
            function_name="plus_dma",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=2 * self.directional_movement_lookback_period,
            min_periods=self.directional_movement_min_periods,
            symmetric=None,
            operation=self._plus_dm,
            operation_args=(self.directional_movement_lookback_period),
        )
        minus_dma = self._minus_dma_object._template_feature_calculation(
            function_name="minus_dma",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=2 * self.directional_movement_lookback_period,
            min_periods=self.directional_movement_min_periods,
            symmetric=None,
            operation=self._minus_dm,
            operation_args=(self.directional_movement_lookback_period),
        )

        names = (
            plus_dma.columns if isinstance(plus_dma, pd.DataFrame) else [plus_dma.name]
        )

        frame_index = plus_dma.index
        plus_dma_frame = pd.DataFrame(
            np.where(((plus_dma > minus_dma) & plus_dma > 0), plus_dma, 0),
            columns=names,
            index=frame_index.values,
        )

        minus_dma_frame = pd.DataFrame(
            np.where(((minus_dma > plus_dma) & minus_dma > 0), minus_dma, 0),
            columns=names,
            index=frame_index.values,
        )

        smoothed_plus_dma = self._plus_dm_smoothing_object.fit(
            dataframe=plus_dma_frame, first_fit=first_fit
        )

        smoothed_minus_dma = self._minus_dm_smoothing_object.fit(
            dataframe=minus_dma_frame,
            first_fit=first_fit,
        )

        average_true_range = self._average_true_range_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
        )
        average_true_range=average_true_range.to_frame() if isinstance(average_true_range,pd.Series) else average_true_range

        plus_directional_index = 100 * (smoothed_plus_dma.div(average_true_range))
        minus_directional_index = 100 * (smoothed_minus_dma.div(average_true_range))

        temp_adx = np.abs(
            (plus_directional_index - minus_directional_index)
            / (plus_directional_index + minus_directional_index)
        )

        adx = self._average_dm_smoothing_object.fit(
            dataframe=temp_adx,
            first_fit=first_fit,
        )
        return adx

df=pd.DataFrame({'a':np.random.random(20)})
ob = AverageDirectionalMovementIndex(
    directional_movement_lookback_period=8,

    directional_movement_smoothing_period=8,

    average_directional_movement_smoothing_period=8,

    true_range_lookback=8,
    average_true_range_span=8,
)
print(ob.fit(df["a"], first_fit=True))