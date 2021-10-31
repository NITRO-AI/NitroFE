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


class ZeroLagExponentialMovingFeature:
    def __init__(
        self,
        lag_period: int = 5,
        alpha: float = None,
        operation: str = "mean",
        initialize_using_operation: bool = False,
        initialize_span: int = None,
        com: float = None,
        span: float = None,
        halflife: float = None,
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
    ):
        """

        Parameters
        ----------
        lag_period : int ,optional
            lag peiod for feature calculation
        alpha : float, optional
            Specify smoothing factor  directly, by default None
        operation : str, {'mean','var','std'}
            operation to be performed for the moving feature,available operations are 'mean','var','std', by default 'mean'
        initialize_using_operation : bool, optional
            If True, then specified 'operation' is performed on the first 'initialize_span' values, and then the exponential moving average is calculated, by default False
        initialize_span : int, optional
            the span over which 'operation' would be performed for initialization, by default None
        com : float, optional
            Specify decay in terms of center of mass, by default None
        span : float, optional
            pecify decay in terms of span , by default None
        halflife : float, optional
            Specify decay in terms of half-life, by default None
        min_periods : int, optional
            Minimum number of observations in window required to have a value (otherwise result is NA)., by default 0
        ignore_na : bool, optional
            Ignore missing values when calculating weights; specify True to reproduce pre-0.15.0 behavior, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None
        """

        self.com = com
        self.lag_period = lag_period
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.min_periods = min_periods
        self.ignore_na = ignore_na
        self.axis = axis
        self.times = times
        self.operation = operation
        self.initialize_using_operation = initialize_using_operation
        self.initialize_span = initialize_span

    def _sub_lag(self, x):
        return 2 * x.iloc[-1] - x.iloc[0]

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
    ):
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
            self._zlema_object = ExponentialMovingFeature(
                com=self.com,
                operation=self.operation,
                span=self.span,
                halflife=self.halflife,
                alpha=self.alpha,
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
                min_periods=self.min_periods,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
            )
            self._lag_object = weighted_window_features()

        res = self._lag_object._template_feature_calculation(
            function_name="_lag_object",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=int((self.lag_period - 1) / 2),
            min_periods=None,
            symmetric=None,
            operation=self._sub_lag,
            operation_args=(),
        )

        res = self._zlema_object.fit(
            dataframe=res,
            first_fit=first_fit,
        )

        return res
