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


class SeriesWeightedMovingFeature:
    def __init__(
        self,
        lookback_period: int = 4,
        min_periods: int = None,
        operation: Callable = None,
        operation_args: tuple = (),
    ):
        """
        Parameters
        ----------
        lookback_period : int, optional
            Size of the rolling window for lookback, by default 4
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default None
        operation : Callable, optional
            operation to perform over values. If None, numpy mean is used, by default None
        operation_args : tuple, optional
            additional agrument values to be sent for operation function
        """
        self.lookback_period = lookback_period
        self.min_periods = min_periods
        self.operation = np.mean if operation==None else operation
        self.operation_args = operation_args

        

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        dataframe_for_weight: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True):
        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values
        dataframe_for_weight : Union[pd.DataFrame, pd.Series]
            dataframe containing column values
        first_fit : bool, optional
            Indicator features require past values for calculation.
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True

        """
        if first_fit:
            self._multiplication_object = weighted_window_features()
            self._weight_object = weighted_window_features()

        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()
        if isinstance(dataframe_for_weight, pd.Series):
            dataframe_for_weight = dataframe_for_weight.to_frame()

        multiplication_res = pd.DataFrame(
            np.multiply(dataframe.values, dataframe_for_weight.values),
            columns=dataframe.columns,
        )

        _multiplication_value = (
            self._multiplication_object._template_feature_calculation(
                function_name="_multiplication_object",
                win_function=_identity_window,
                first_fit=first_fit,
                dataframe=multiplication_res,
                window=self.lookback_period,
                min_periods=self.min_periods,
                symmetric=None,
                operation=self.operation,
                operation_args=self.operation_args,
            )
        )

        _weight_value = self._multiplication_object._template_feature_calculation(
            function_name="_weight_value",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe_for_weight,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self.operation,
            operation_args=self.operation_args,
        )

        res = pd.DataFrame(
            (_multiplication_value.values / _weight_value.values),
            columns=dataframe.columns,
        )

        return res
