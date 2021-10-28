

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
from NitroFE.time_based_features.moving_average_features.moving_average_features import ExponentialMovingFeature,HullMovingFeature,\
    KaufmanAdaptiveMovingAverage,FractalAdaptiveMovingAverage,TripleExponentialMovingFeature,SmoothedMovingAverage


class RelativeStrengthIndex:
    def __init__(self,
        lookback_period:int=8):
        """
        Parameters
        ----------
        lookback_period : int, optional
            Size of the rolling window for lookback, by default 8
        """
        self.lookback_period = lookback_period

    def _diff_pos(self,x): 
        diff_val=x.iloc[-1]-x.iloc[-2]
        return diff_val if diff_val>0 else 0 

    def _diff_neg(self,x): 
        diff_val=x.iloc[-1]-x.iloc[-2]
        res=diff_val if diff_val<0 else 0
        return -res

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True):
        """

        Provided dataframe must be in ascending order.

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
            self._up_object = weighted_window_features()
            self._down_object = weighted_window_features()

            self._up_smoothed=SmoothedMovingAverage(lookback_period=self.lookback_period)
            self._down_smoothed=SmoothedMovingAverage(lookback_period=self.lookback_period)

        
        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()

        up_value = self._up_object._template_feature_calculation(
            function_name="_up_object",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=2 ,
            min_periods=None,
            symmetric=None,
            operation=self._diff_pos,
            operation_args=(),
        )

        down_value = self._down_object._template_feature_calculation(
            function_name="_down_object",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=2 ,
            min_periods=None,
            symmetric=None,
            operation=self._diff_neg,
            operation_args=(),
        )

        smoothed_up_value=self._up_smoothed.fit(dataframe=up_value,first_fit=first_fit)
        smoothed_down_value=self._down_smoothed.fit(dataframe=down_value,first_fit=first_fit)

        rsi =  100 -100/(1+(smoothed_up_value/smoothed_down_value))
        return rsi     