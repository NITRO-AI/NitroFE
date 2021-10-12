import numpy as np
import pandas as pd
from typing import Union, Callable

from pandas.core.frame import DataFrame
from NitroFE.time_based_features.moving_average_features.moving_average_features import (
    exponential_moving_feature,
)
from NitroFE.time_based_features.weighted_window_features.weighted_window_features import (
    weighted_window_features,
)
from weighted_windows import _equal_window, _identity_window


class absolute_price_oscillator:
    def __init__(self):
        pass

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        fast_period: int = 2,
        slow_period: int = 5,
        fast_operation: str = "mean",
        slow_operation: str = "mean",
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
    ):
        """calculate_absolute_price_oscillator

        The Absolute Price Oscillator displays the difference between two exponential moving averages,
        of a security's price and is expressed as an absolute value.

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        fast_period : int, optional
            specify decay in terms of span, for the fast moving feature , by default 10
        slow_period : int, optional
            specify decay in terms of span, for the slow moving feature, by default 20
        fast_operation : str, optional
            operation to be performed for the fast moving feature, by default 'mean'
        slow_operation : str, optional
            operation to be performed for the slow moving feature, by default 'mean'
        ignore_na : bool, optional
            gnore missing values when calculating weights, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None

        Returns
        -------
        absolute price oscillator

        References
        -----
        .. [1] fidelity, "apo",
            https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo
        """

        if first_fit:
            self._fast_em_object = exponential_moving_feature()
            self._slow_em_object = exponential_moving_feature()

            self.span_fast = fast_period
            self.span_slow = slow_period

            self.min_periods = min_periods

            self.ignore_na = ignore_na
            self.axis = axis
            self.times = times
            self.fast_operation = fast_operation
            self.slow_operation = slow_operation

        fast_em = self._fast_em_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
            span=self.span_fast,
            ignore_na=self.ignore_na,
            axis=self.axis,
            times=self.times,
            operation=self.fast_operation,
        )
        slow_em = self._slow_em_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
            span=self.span_slow,
            ignore_na=self.ignore_na,
            axis=self.axis,
            times=self.times,
            operation=self.slow_operation,
        )

        absolute_price_oscillator = fast_em - slow_em
        return absolute_price_oscillator


class moving_average_convergence_divergence:
    def __init__(self):
        pass

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        fast_period: int = 12,
        slow_period: int = 26,
        smoothing_period: int = 9,
        fast_operation: str = "mean",
        slow_operation: str = "mean",
        smoothing_operation: str = "mean",
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
        return_histogram=False,
    ):

        """calculate_moving_average_convergence_divergence [summary]

        Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential
        moving average (EMA) from the 12-period EMA.

        The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line,"
        is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals.

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        fast_period : int, optional
            specify decay in terms of span, for the fast moving feature, by default 12
        slow_period : int, optional
            specify decay in terms of span, for the slow moving feature, by default 26
        smoothing_period : int, optional
            specify decay in terms of span, for the smoothing moving feature, by default 9
        fast_operation : str, {'mean','var','std'}
            operation to be performed for the fast moving feature, by default 'mean'
        slow_operation : str, {'mean','var','std'}
            operation to be performed for the slow moving feature, by default 'mean'
        smoothing_operation : str, optional
            operation to be performed for the smoothing moving feature, by default 'mean'
        ignore_na : bool, optional
            gnore missing values when calculating weights, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None
        return_histogram : bool, optional
            if True returns histogram values instead of smoothed signal values, by default False

        Returns
        -------
        Smoothed signal line , macd histogram

        References
        -----
        .. [1] investopedia, "Moving Average Convergence Divergence (MACD)",
            https://www.investopedia.com/terms/m/macd.asp#:~:text=The%20MACD%20is%20calculated%20by,for%20buy%20and%20sell%20signals.
        """
        if first_fit:
            self._raw_macd_object = absolute_price_oscillator()
            self._macd_object = exponential_moving_feature()

            self.span_fast = fast_period
            self.span_slow = slow_period
            self.min_periods = min_periods
            self.ignore_na = ignore_na
            self.axis = axis
            self.times = times
            self.fast_operation = fast_operation
            self.slow_operation = slow_operation
            self.smoothing_operation = smoothing_operation
            self.smoothing_period = smoothing_period
            self.return_histogram = return_histogram

        raw_macd = self._raw_macd_object.fit(
            dataframe,
            first_fit=first_fit,
            fast_period=self.span_fast,
            slow_period=self.span_slow,
            fast_operation=self.fast_operation,
            slow_operation=self.slow_operation,
            min_periods=self.min_periods,
            ignore_na=self.ignore_na,
            axis=self.axis,
            times=self.times,
        )

        macd = self._macd_object.fit(
            dataframe=raw_macd,
            first_fit=first_fit,
            span=self.smoothing_period,
            ignore_na=self.ignore_na,
            axis=self.axis,
            times=self.times,
            operation=self.smoothing_operation,
        )

        return raw_macd - macd if self.return_histogram else macd


class Average_true_range:
    def __init__(self):
        pass

    def true_range(self, x):
        return np.max(
            [
                (np.max(x) - np.min(x)),
                np.abs(np.max(x) - x.iloc[-1]),
                np.abs(np.min(x) - x.iloc[-1]),
            ]
        )

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        true_range_lookback: int = 4,
        average_true_range_span: int = 6,
        true_range_min_periods: int = None,
        average_true_range_periods: int = 1,
    ):
        """
        Average true range is a  volatility indicator.The indicator does not provide an indication of price trend,
        simply the degree of volatility.The average true range is an N-period smoothed moving average of the true range values.

        first_range= (Highest value in `true_range_lookback` rolling period ) - (Lowest value in `true_range_lookback` rolling period)
        second_range= Absolute((Highest value in `true_range_lookback` rolling period ) - (Latest value in `true_range_lookback` rolling period))
        third_range= Absolute((Lowest value in `true_range_lookback` rolling period ) - (Latest value in `true_range_lookback` rolling period))

        True_Range = Max[ first_range, second_range, third_range ]

        Average_True_range = Mean of True_Range over `average_true_range_periods` rolling period

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        true_range_lookback : int, optional
            Size of the rolling window for true range value calculation, by default 4
        average_true_range_span : int, optional
            Size of the rolling window for average true range value calculation, by default 6
        true_range_min_periods : int, optional
            Minimum number of observations in window required to have a value for true range calculation, by default None
        average_true_range_periods : int, optional
            Minimum number of observations in window required to have a value for average true range calculation , by default 1

        Returns
        -------
        Smoothed signal line , macd histogram

        References
        -----
        .. [1] investopedia, "Average True Range (ATR)",
            https://www.investopedia.com/terms/a/atr.asp
        """
        if first_fit:
            self.true_range_lookback = true_range_lookback
            self.average_true_range_span = average_true_range_span

            self.true_range_min_periods = true_range_min_periods
            self.average_true_range_periods = average_true_range_periods

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


class average_directional_movement_index:
    def __init__(self):
        pass

    def _plus_dm(self, x, look_back_period):
        return np.max(x.iloc[look_back_period:]) - np.max(
            x.iloc[0 : look_back_period - 1]
        )

    def _minus_dm(self, x, look_back_period):
        return np.min(x.iloc[0 : look_back_period - 1]) - np.min(
            x.iloc[look_back_period:]
        )

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        directional_movement_lookback_period: int = 4,
        directional_movement_min_periods: int = None,
        directional_movement_smoothing_period: int = 14,
        directional_movement_smoothing_min_periods: int = None,
        average_directional_movement_smoothing_period: int = 14,
        average_directional_movement_min_periods: int = None,
    ):
        """average_directional_movement_index


        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        directional_movement_lookback_period : int, optional
            [description], by default 4
        directional_movement_min_periods : int, optional
            [description], by default None
        directional_movement_smoothing_period : int, optional
            [description], by default 14
        directional_movement_smoothing_min_periods : int, optional
            [description], by default None
        average_directional_movement_smoothing_period : int, optional
            [description], by default 14
        average_directional_movement_min_periods : int, optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        if first_fit:

            self._plus_dma_object = weighted_window_features()
            self._minus_dma_object = weighted_window_features()

            self._plus_dm_smoothing_object = exponential_moving_feature()
            self._minus_dm_smoothing_object = exponential_moving_feature()

            self._average_true_range_object = Average_true_range()

            self._average_dm_smoothing_object = exponential_moving_feature()

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
            plus_dma.columns if isinstance(plus_dma, pd.DataFrame) else plus_dma.name
        )
        frame_index = plus_dma.index
        plus_dma_frame = pd.DataFrame(
            np.where(((plus_dma > minus_dma) & plus_dma > 0), plus_dma, 0),
            columns=names,
            index=frame_index,
        )

        minus_dma_frame = pd.DataFrame(
            np.where(((minus_dma > plus_dma) & minus_dma > 0), minus_dma, 0),
            columns=names,
            index=frame_index,
        )

        smoothed_plus_dma = self._plus_dm_smoothing_object.fit(
            dataframe=plus_dma_frame,
            first_fit=first_fit,
            alpha=1 / (self.directional_movement_smoothing_period),
            min_periods=self.directional_movement_smoothing_min_periods,
            operation="mean",
        )

        smoothed_minus_dma = self._minus_dm_smoothing_object.fit(
            dataframe=minus_dma_frame,
            first_fit=first_fit,
            alpha=1 / (self.directional_movement_smoothing_period),
            min_periods=self.directional_movement_smoothing_min_periods,
            operation="mean",
        )

        average_true_range = self._average_true_range_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
            true_range_lookback=4,
            average_true_range_span=6,
            true_range_min_periods=1,
            average_true_range_periods=1,
        )

        plus_directional_index = 100 * smoothed_plus_dma.div(average_true_range)
        minus_directional_index = 100 * smoothed_minus_dma.div(average_true_range)

        temp_adx = np.abs(
            (plus_directional_index - minus_directional_index)
            / (plus_directional_index + minus_directional_index)
        )

        adx = self._average_dm_smoothing_object.fit(
            dataframe=temp_adx,
            first_fit=first_fit,
            alpha=1 / (self.directional_movement_smoothing_period),
            min_periods=self.directional_movement_smoothing_min_periods,
            operation="mean",
        )
        return adx


#####################################################################

# ob = average_directional_movement_index()
# df = pd.DataFrame({"a": np.arange(25), "b": np.arange(25) + 2})
# df = pd.DataFrame(
#     {
#         "a": 10 * np.random.random(25),
#         "b": np.arange(0, 100, step=2)[:25]   + np.random.choice(np.arange(-20,20), size=25),
#     }
# )
# wz = 4
# mp = 2
# flp = 15
# print("df", df)
# res_all = ob.fit(
#                 df,
#                 first_fit=True,
#             directional_movement_lookback_period = wz,
#             directional_movement_min_periods = 1,

#             directional_movement_smoothing_period= 14,
#             directional_movement_smoothing_min_periods= 1,

#             average_directional_movement_smoothing_period= 14,
#             average_directional_movement_min_periods= 1,
#             )
# print('res_all',res_all)

# res_comb = pd.concat(
#     [
#         ob.fit(
#             df.iloc[:flp],
#             first_fit=True,
#             directional_movement_lookback_period = wz,
#             directional_movement_min_periods = 1,

#             directional_movement_smoothing_period= 14,
#             directional_movement_smoothing_min_periods= 1,

#             average_directional_movement_smoothing_period= 14,
#             average_directional_movement_min_periods= 1,
#         ),
#         ob.fit(df.iloc[flp:], first_fit=False),
#     ]
# )
# print(res_comb)


# print(pd.concat([res_all, res_comb], axis=1))
# print((res_all.fillna(0)!=res_comb.fillna(0)).sum())
#####################################################################
