import numpy as np
import pandas as pd
from typing import Union, Callable

from pandas.core.frame import DataFrame

from NitroFE.time_based_features.moving_average_features.moving_average_features import (
    exponential_moving_feature,
    triple_exponential_moving_feature,
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

        Absolute Price Oscillator = (Fast exponential moving average) - (Slow exponential moving average)

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
            Ignore missing values when calculating weights, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None

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

        """
        calculate_moving_average_convergence_divergence

        Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential
        moving average (EMA) from the 12-period EMA.

        The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line,"
        is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals.

        Moving average convergence divergence (MACD) = Exponential moving average of (Absolute Price)

        Moving average convergence divergence histogram = (Absolute Price)- (Exponential moving average of (Absolute Price))

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
            Ignore missing values when calculating weights, by default False
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
        true_range_lookback: int = 4,
        average_true_range_span: int = 6,
        true_range_min_periods: int = None,
        average_true_range_periods: int = 1,
    ):
        """
        average_directional_movement_index

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

        References
        -----
        .. [1] wikipedia, "Average directional movement index",
            https://en.wikipedia.org/wiki/Average_directional_movement_index

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
            self.true_range_lookback = true_range_lookback
            self.average_true_range_span = average_true_range_span
            self.true_range_min_periods = true_range_min_periods
            self.average_true_range_periods = average_true_range_periods

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
            true_range_lookback=true_range_lookback,
            average_true_range_span=average_true_range_span,
            true_range_min_periods=true_range_min_periods,
            average_true_range_periods=average_true_range_periods,
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
            alpha=1 / (self.average_directional_movement_smoothing_period),
            min_periods=self.average_directional_movement_min_periods,
            operation="mean",
        )
        return adx


class aroon:
    def __init__(self):
        pass

    def _calculate_aroon_up(self, x, look_back_period):
        return x.argmax() / (look_back_period)

    def _calculate_aroon_down(self, x, look_back_period):
        return x.argmin() / (look_back_period)

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        lookback_period: int = 4,
        min_periods: int = None,
    ):
        """
        Aroon

        Aroon Oscillator = (Aroon Up) − (Aroon Down)
        Aroon Up = 100*(lookback_period - Periods Since lookback_period-High)/(lookback_period)
        Aroon Down = 100*(lookback_period - Periods Since lookback_period-Low)/(lookback_period)

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        lookback_period : int, optional
            Size of the rolling window for lookback, by default 4
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default None

        References
        -----
        .. [1] investopedia, "Aroon",
            https://www.investopedia.com/terms/a/aroonoscillator.asp

        """

        if first_fit:
            self._aroon_up_object = weighted_window_features()
            self._aroon_down_object = weighted_window_features()
            self.lookback_period = lookback_period
            self.min_periods = min_periods

        aroon_up = self._aroon_up_object._template_feature_calculation(
            function_name="aroon_up",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._calculate_aroon_up,
            operation_args=(self.lookback_period,),
        )
        aroon_down = self._aroon_up_object._template_feature_calculation(
            function_name="aroon_down",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._calculate_aroon_down,
            operation_args=(self.lookback_period,),
        )
        aroon_value = 100 * (aroon_up - aroon_down)

        return aroon_value


class typical_value:
    def __init__(self):
        pass

    def _calculate_typical_value(self, x):
        return (np.max(x) + np.min(x) + x.iloc[-1:]) / 3

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        lookback_period: int = 6,
        min_periods: int = None,
    ):
        """
        typical_value

        Highest = Highest value in the 'lookback_period'
        Lowest = Highest value in the 'lookback_period'
        Current =  Latest value in the 'lookback_period'

        Typical Value = ( Highest + Lowest + Current )/3

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        lookback_period : int, optional
            Size of the rolling window for lookback, by default 6
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default None

        """

        if first_fit:
            self._typical_value_object = weighted_window_features()
            self.lookback_period = lookback_period
            self.min_periods = min_periods

        _typical_value = self._typical_value_object._template_feature_calculation(
            function_name="typical_value",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._calculate_typical_value,
            operation_args=(),
        )

        return _typical_value


class bollinger_bands:
    def __init__(self):
        pass

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        typical_value_lookback_period: int = 6,
        typical_value_min_periods: int = None,
        moving_average_typical_value_lookback_period: int = 6,
        moving_average_typical_value_min_periods: int = None,
        standard_deviation_multiplier: int = 2,
    ):
        """
        bollinger_bands

        Highest = Highest value in the 'lookback_period'
        Lowest = Highest value in the 'lookback_period'
        Current =  Latest value in the 'lookback_period'

        Typical Value = ( Highest + Lowest + Current )/3

        MA of typical value = Moving average of typical value with 'moving_average_typical_value_lookback_period' window size
        STD of typical value = Standard deviation of typical value with 'moving_average_typical_value_lookback_period' window size

        Positive band = MA of typical value + 'standard_deviation_multiplier' * STD of typical value
        Negative band = MA of typical value - 'standard_deviation_multiplier' * STD of typical value

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
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

        References
        -----
        .. [1] wikipedia, "Bollinger_Bands",
            https://en.wikipedia.org/wiki/Bollinger_Bands
        """

        if first_fit:
            self._typical_value_object = typical_value()
            self._ma_bollinger_bands_object = weighted_window_features()
            self._std_bollinger_bands_object = weighted_window_features()

            self.typical_value_lookback_period = typical_value_lookback_period
            self.typical_value_min_periods = typical_value_min_periods

            self.moving_average_typical_value_lookback_period = (
                moving_average_typical_value_lookback_period
            )
            self.moving_average_typical_value_min_periods = (
                moving_average_typical_value_min_periods
            )

            self.standard_deviation_multiplier = standard_deviation_multiplier

        _typical_value = self._typical_value_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
            lookback_period=self.typical_value_lookback_period,
            min_periods=self.typical_value_min_periods,
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


class kaufman_efficiency:
    def __init__(self):
        pass

    def _calculate_kaufman_efficiency(self, x):
        up = np.abs(x.iloc[-1] - x.iloc[0])
        down = np.abs(x.diff().fillna(0)).sum()
        return up / down

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        lookback_period: int = 4,
        min_periods: int = None,
    ):
        """
        kaufman_efficiency

        It is calculated by dividing the net change in price movement over 'lookback_period' periods,
         by the sum of the absolute net changes over the same 'lookback_period' periods.

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        lookback_period : int, optional
            Size of the rolling window for lookback, by default 4
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default None

        References
        -----
        .. [1] strategyquant, "Kaufman’s Efficiency Ratio",
            https://strategyquant.com/codebase/kaufmans-efficiency-ratio-ker/
        """

        if first_fit:
            self._kaufman_efficiency_object = weighted_window_features()
            self.lookback_period = lookback_period
            self.min_periods = min_periods

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


class triple_exponential_moving_average_oscillator:
    def __init__(self):
        pass

    def _ocs_value(self, x):
        return (x.iloc[-1] - x.iloc[0]) / x.iloc[0]

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        com: float = None,
        operation: str = "mean",
        span: float = None,
        halflife: float = None,
        alpha: float = None,
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
    ):
        """
        triple_exponential_moving_average_oscillator

        Percentage change of triple exponential moving average

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        com : float, optional
            Specify decay in terms of center of mass, by default None
        span : float, optional
            pecify decay in terms of span , by default None
        halflife : float, optional
            Specify decay in terms of half-life, by default None
        alpha : float, optional
            Specify smoothing factor  directly, by default None
        min_periods : int, optional
            Minimum number of observations in window required to have a value (otherwise result is NA)., by default 0
        adjust : bool, optional
            Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings (viewing EWMA as a moving average)
            , When adjust=False, the exponentially weighted function is calculated recursively,by default False
        ignore_na : bool, optional
            Ignore missing values when calculating weights; specify True to reproduce pre-0.15.0 behavior, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None
        operation : str, {'mean','var','std'}
            operation to be performed for the moving feature, by default 'mean'

        References
        -----
        .. [1] investopedia, "triple-exponential-moving-average",
            https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp
        """

        if first_fit:
            self._osc_object = triple_exponential_moving_feature()
            self._difference_object = weighted_window_features()

        res = self._osc_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
            com=com,
            operation=operation,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            ignore_na=ignore_na,
            axis=axis,
            times=times,
        )

        res = self._difference_object._template_feature_calculation(
            function_name="triple_exponential_moving_average_oscillator",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=res,
            window=2,
            min_periods=None,
            symmetric=None,
            operation=self._ocs_value,
            operation_args=(),
        )
        return res


class zero_lag_exponential_moving_feature:
    def __init__(self):
        pass

    def _sub_lag(self, x):
        return x.iloc[0] - x.iloc[-1]

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        lag_period: int = 5,
        com: float = None,
        operation: str = "mean",
        span: float = None,
        halflife: float = None,
        alpha: float = None,
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
    ):
        """
        Zero lag exponential moving average

        Lag = ( 'lag_period' - 1 )/2
        EmaData = Data + (Data -Data('Lag' periods ago))
        ZLEMA = Exponential moving average of 'EMAData'


        Provided dataframe must be in ascending order.

        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create exponential moving feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        lag_period : int , optional
            lag period length , by default 5
        com : float, optional
            Specify decay in terms of center of mass, by default None
        span : float, optional
            pecify decay in terms of span , by default None
        halflife : float, optional
            Specify decay in terms of half-life, by default None
        alpha : float, optional
            Specify smoothing factor  directly, by default None
        min_periods : int, optional
            Minimum number of observations in window required to have a value (otherwise result is NA)., by default 0
        ignore_na : bool, optional
            Ignore missing values when calculating weights; specify True to reproduce pre-0.15.0 behavior, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None
        operation : str, {'mean','var','std'}
            operation to be performed for the moving feature, by default 'mean'

        References
        -----
        .. [1] wikipedia, "Zero lag exponential moving average",
            https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
        """
        if first_fit:
            self.lag_period = lag_period
            self._zlema_object = exponential_moving_feature()
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
            com=com,
            operation=operation,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            ignore_na=ignore_na,
            axis=axis,
            times=times,
        )

        return res


class series_weighted_average:
    def __init__(self):
        pass

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        dataframe_for_weight: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
    ):
        """
        series_weighted_average
        This feature is an implementation of volume weighted moving average

        series_weighted_average = Summation( dataframe * dataframe_for_weight )/ Summation( dataframe_for_weight )


        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values
        dataframe_for_weight : Union[pd.DataFrame, pd.Series]
            dataframe containing column values
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True

        References
        -----
        .. [1] investopedia, "Volume-Weighted Average Price",
            https://www.investopedia.com/terms/v/vwap.asp
        """
        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()
        if isinstance(dataframe_for_weight, pd.Series):
            dataframe_for_weight = dataframe_for_weight.to_frame()

        multiplication_res = (
            pd.DataFrame(np.multiply(dataframe.values,dataframe_for_weight.values),columns=dataframe.columns)
            if first_fit
            else pd.concat(
                [
                    self.multiplication_values_from_last_run,
                    pd.DataFrame(np.multiply(dataframe.values,dataframe_for_weight.values),columns=dataframe.columns),
                ]
            )
        )

        if not first_fit:
            dataframe_for_weight = pd.concat(
                [self.values_from_last_run, dataframe_for_weight]
            )

        cumilative_res = dataframe_for_weight.expanding().sum()

        cumilative_multiplication_res = multiplication_res.expanding().sum()
        self.multiplication_values_from_last_run = cumilative_multiplication_res.iloc[
            -1:
        ]

        cumilative_multiplication_res = (
            cumilative_multiplication_res.iloc[1:]
            if (not first_fit)
            else cumilative_multiplication_res
        )
        cumilative_res = cumilative_res.iloc[1:] if (not first_fit) else cumilative_res
        dataframe_for_weight = (
            dataframe_for_weight.iloc[1:] if (not first_fit) else dataframe_for_weight
        )

        res = pd.DataFrame(cumilative_multiplication_res.values/(cumilative_res.values))
        self.values_from_last_run = cumilative_res.iloc[-1:]

        return res


class elastic_series_weighted_average:
    def __init__(self):
        pass

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        dataframe_for_weight: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        weight_sum_lookback: int = 4):
        """
        elastic_series_weighted_average
        This feature is an implementation of elastic volume weighted moving average

        L_eswa= Previous 'elastic_series_weighted_average' value
        w = current value of 'dataframe_for_weight'
        Wn = Sum of last 'weight_sum_lookback' values of dataframe_for_weight
        x = Current 'dataframe' value
        elastic_series_weighted_average =  L_eswa + ( w/Wn ) * ( x - L_eswa )

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values
        dataframe_for_weight : Union[pd.DataFrame, pd.Series]
            dataframe containing column values
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        weight_sum_lookback : int, optional
            Size of the rolling window of lookback , by default 4

        """

        if first_fit:
            self._object = weighted_window_features()
            self.weight_sum_lookback = weight_sum_lookback

        rolling_sum = self._object._template_feature_calculation(
            function_name="_lag_object",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe_for_weight,
            window=self.weight_sum_lookback,
            min_periods=1,
            symmetric=None,
            operation=np.sum,
            operation_args=(),
        ).fillna(0)

        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()
        
        eswa = pd.DataFrame(np.zeros(dataframe.shape), columns=dataframe.columns)
        print(dataframe.shape)
        print(eswa)
        if first_fit:
            eswa = pd.concat([dataframe.iloc[:1], eswa])
        else:
            eswa = pd.concat([self.values_from_last_run, eswa])

        eswa["_iloc"] = np.arange(len(eswa))
        ll = [x for x in eswa.columns if x != "_iloc"]

        for r1, r2, r3, r4 in zip(
            dataframe.iterrows(),
            eswa[1:].iterrows(),
            dataframe_for_weight.iterrows(),
            rolling_sum.iterrows(),
        ):

            previous_eswa = eswa[eswa["_iloc"] == (r2[1]["_iloc"] - 1)][ll]
            eswa.loc[eswa["_iloc"] == r2[1]["_iloc"], ll] = (
                previous_eswa.fillna(0)
                + ((r1[1] - previous_eswa) * (r3[1] / r4[1])).fillna(0)
            ).values[0]

        
        res=eswa.iloc[1:][ll]
        
        self.values_from_last_run = eswa.iloc[-1:]
        return res


#####################################################################


# ob = elastic_series_weighted_average()
# df = pd.DataFrame(
#     {
#         "a": 10 * np.random.random(10),
#         "b": np.arange(0, 100, step=2)[:10]
#         + np.random.choice(np.arange(0, 20), size=10),
#     }
# )
# df2 = pd.DataFrame(
#     {
#         "a": 10 * np.random.random(10),
#         "b": np.arange(0, 100, step=2)[:10]
#         + np.random.choice(np.arange(0, 20), size=10),
#     }
# )
# #df=df['a']
# #df2=df2['a']
# #df=pd.Series(10 * np.random.random(10))
# #df2=pd.Series(10 * np.random.random(10))
# wz = 6
# mp = 1
# flp = 5
# flp2 = 7
# kw = {}
# #print("df", df)
# res_all = ob.fit(df[:], df2[:], first_fit=True, **kw).reset_index(drop=True)
# print("res_all", res_all)
# print("#####################################################################")
# res_comb = pd.concat(
#     [
#         ob.fit(df.iloc[:flp], df2.iloc[:flp], first_fit=True, **kw),
#         ob.fit(df.iloc[flp:flp2], df2.iloc[flp:flp2], first_fit=False),
#         ob.fit(df.iloc[flp2:], df2.iloc[flp2:], first_fit=False),
#     ]
# ).reset_index(drop=True)
# print('res_comb',res_comb)
# print(pd.concat([res_all, res_comb], axis=1))
# print("#####################################################################")
# print((res_all.fillna(0) != res_comb.fillna(0)).sum())


#####################################################################
