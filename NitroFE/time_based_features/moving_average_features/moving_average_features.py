import pandas as pd
import numpy as np
from typing import Union, Callable

from NitroFE.time_based_features.weighted_window_features.weighted_window_features import (
    weighted_window_features,
)
from NitroFE.time_based_features.weighted_window_features.weighted_windows import (
    _equal_window,
    _identity_window,
)


class ExponentialMovingFeature:
    """
    Provided dataframe must be in ascending order.
    """

    def __init__(
        self,
        alpha: float = None,
        operation: str = "mean",
        initialize_using_operation: bool = False,
        initialize_span: int = None,
        com: float = None,
        span: int = None,
        halflife: float = None,
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
    ):
        """
        Parameters
        ----------
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
            specify decay in terms of span , by default None
        halflife : float, optional
            Specify decay in terms of half-life, by default None
        min_periods : int, optional
            Minimum number of observations in window required to have a value (otherwise result is NA), by default 0
        ignore_na : bool, optional
            Ignore missing values when calculating weights; specify True to reproduce pre-0.15.0 behavior, by default False
        axis : int, optional
            The axis to use. The value 0 identifies the rows, and 1 identifies the columns, by default 0
        times : str, optional
            Times corresponding to the observations. Must be monotonically increasing and datetime64[ns] dtype, by default None
        """

        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.min_periods = min_periods
        self.adjust = False
        self.ignore_na = ignore_na
        self.axis = axis
        self.times = times
        self.operation = operation
        self.last_values_from_previous_run = None
        self.initialize_using_operation = initialize_using_operation
        self.initialize_span = initialize_span

    def _perform_temp_operation(self, x):
        _return = (
            x.mean()
            if self.operation == "mean"
            else x.var()
            if self.operation == "var"
            else x.std()
            if self.operation == "std"
            else None
        )
        if _return is None:
            raise ValueError(f"Operation {self.operation} not supported")
        return _return

    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], first_fit: bool = True):
        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create exponential moving feature over
        first_fit : bool, optional
            Moving features require past values for calculation.
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        """
        if not first_fit:
            if self.last_values_from_previous_run is None:
                raise ValueError(
                    "First fit has not occured before. Kindly run first_fit=True for first fit instance,"
                    "and then proceed with first_fit=False for subsequent fits "
                )
            self.adjust = False
            dataframe = pd.concat(
                [self.last_values_from_previous_run, dataframe], axis=0
            )
        else:

            if self.initialize_using_operation:
                self.min_periods = None
                if (self.initialize_span is None) and (self.span is None):
                    raise ValueError(
                        "For initialize_using_operation=True,"
                        "either initialize_span or span value is required"
                    )
                elif (self.initialize_span is None) and (self.span is not None):
                    self.initialize_span = self.span

                first_frame = self._perform_temp_operation(
                    dataframe[: self.initialize_span].rolling(
                        window=self.initialize_span
                    )
                )
                dataframe = pd.concat([first_frame, dataframe[self.initialize_span :]])
            else:
                if self.initialize_span is not None:
                    raise ValueError(
                        "In order to use initialize_span, initialize_using_operation must be True"
                    )

        _dataframe = dataframe.ewm(
            com=self.com,
            span=self.span,
            halflife=self.halflife,
            alpha=self.alpha,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            axis=self.axis,
            times=self.times,
        )
        _return = (
            _dataframe.mean()
            if self.operation == "mean"
            else _dataframe.var()
            if self.operation == "var"
            else _dataframe.std()
            if self.operation == "std"
            else None
        )
        if _return is None:
            raise ValueError(f"Operation {self.operation} not supported")

        if not first_fit:
            _return = _return.iloc[1:]
        self.last_values_from_previous_run = _return.iloc[-1:]
        return _return


class HullMovingFeature:
    """
    Provided dataframe must be in ascending order.
    """

    def __init__(
        self, window: int = 4, min_periods: int = 1, operation: Callable = None
    ):
        """
        Parameters
        ----------
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        operation : Callable, optional
            operation to perform over the weighted rolling window values, when None is passed, np.mean is used
        """

        self.window = window
        self.min_periods = min_periods

        operation = np.mean if operation == None else operation
        self.operation = operation

        if self.window <= 1:
            raise ValueError(f"window size less than equal to 1 not supported")

        self.window_by_two, self.window_square_root = int(
            np.ceil(self.window / 2)
        ), int(np.ceil(np.sqrt(self.window)))

    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], first_fit: bool = True):
        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which feature is to be constructed
        first_fit : bool, optional
            Moving features require past values for calculation.
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for any subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True

        """
        if first_fit:
            self._window_size_weighted_moving_average_object = (
                weighted_window_features()
            )
            self._window_by_two_size_weighted_moving_average_object = (
                weighted_window_features()
            )
            self._hma_object = weighted_window_features()

        window_size_weighted_moving_average = self._window_size_weighted_moving_average_object.caluclate_weighted_moving_window_feature(
            dataframe=dataframe,
            first_fit=first_fit,
            window=self.window,
            min_periods=self.min_periods,
            operation=self.operation,
        )

        window_by_two_size_weighted_moving_average = self._window_by_two_size_weighted_moving_average_object.caluclate_weighted_moving_window_feature(
            dataframe=dataframe,
            first_fit=first_fit,
            window=self.window_by_two,
            min_periods=self.min_periods,
            operation=self.operation,
        )

        raw_hma = (
            2 * window_by_two_size_weighted_moving_average
            - window_size_weighted_moving_average
        )
        hma = self._hma_object.caluclate_weighted_moving_window_feature(
            dataframe=raw_hma,
            first_fit=first_fit,
            window=self.window_square_root,
            min_periods=self.min_periods,
            operation=self.operation,
        )

        return hma


class _a_kaufman_efficiency:
    def __init__(self):
        pass

    def _calculate_kaufman_efficiency(self, x):
        up = np.abs(x.iloc[-1] - x.iloc[0])
        down = np.abs(x.diff().fillna(0)).sum()
        if down == 0:
            return 0
        else:
            return up / down

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        lookback_period: int = 4,
        min_periods: int = None,
    ):

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


class KaufmanAdaptiveMovingAverage:
    """
    Provided dataframe must be in ascending order.
    """

    def __init__(
        self,
        kaufman_efficiency_lookback_period: int = 4,
        kaufman_efficiency_min_periods: int = None,
        fast_ema_span: int = 2,
        slow_ema_span: int = 5,
    ):
        """
        Parameters
        ----------
        kaufman_efficiency_lookback_period : int, optional
            Size of the rolling window of lookback for the caluculation of kaufman efficiency ratio , by default 4
        kaufman_efficiency_min_periods : int, optional
            Minimum number of observations in window required to have a value for kaufman efficiency ratio, by default 1
        fast_ema_span : int, optional
            fast span length, by default 2
        slow_ema_span : int, optional
            slow span length, by default 5
        """

        self.kaufman_efficiency_lookback_period = kaufman_efficiency_lookback_period
        self.kaufman_efficiency_min_periods = kaufman_efficiency_min_periods
        self.fast_ema_span = fast_ema_span
        self.slow_ema_span = slow_ema_span


    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], first_fit: bool = True):
        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Moving features require past values for calculation.
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        """

        if first_fit:
            self._kaufman_object = _a_kaufman_efficiency()

        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()

        kma = pd.DataFrame(
            np.zeros(dataframe.shape), columns=dataframe.columns, index=dataframe.index
        )
        if first_fit:
            kma = pd.concat(
                [pd.DataFrame(np.zeros((1, kma.shape[1])), columns=kma.columns), kma]
            )
            if self.kaufman_efficiency_min_periods>1:
                _first_pervious=(self.kaufman_efficiency_lookback_period-2) if self.kaufman_efficiency_min_periods==None else (self.kaufman_efficiency_min_periods-2)
            else:
                _first_pervious=0

        else:
            kma = pd.concat([self.values_from_last_run, kma])
            _first_pervious=-1
        kma["_iloc"] = np.arange(len(kma))

        _kaufman_efficiency = self._kaufman_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
            lookback_period=self.kaufman_efficiency_lookback_period,
            min_periods=self.kaufman_efficiency_min_periods,
        )

        ll = [x for x in kma.columns if x != "_iloc"]
        SC = (
            _kaufman_efficiency.copy()
            * (2 / (self.fast_ema_span + 1) - 2 / (self.slow_ema_span + 1))
            + 2 / (self.slow_ema_span + 1)
        )** 2

        if first_fit:
            SC.iloc[_first_pervious]=[0]*SC.shape[1]
 
        for r1, r2, r3 in zip(dataframe[(_first_pervious+1):].iterrows(), kma[(1+_first_pervious+1):].iterrows(), SC[(_first_pervious+1):].iterrows()):

            previous_kama = kma[kma["_iloc"] == (r2[1]["_iloc"] - 1)][ll]

            kma.loc[kma["_iloc"] == r2[1]["_iloc"], ll] = (
                previous_kama
                + np.multiply(r3[1].values, (r1[1].values - previous_kama.values))
            ).values[0]

        
        res = kma.iloc[1:][ll]
        
        self.values_from_last_run = res.iloc[-1:]

        return res


class FractalAdaptiveMovingAverage:
    """
    Provided dataframe must be in ascending order.
    """

    def __init__(self, lookback_period: int = 8, min_periods: int = 1):
        """
        Parameters
        ----------
        lookback_period : int, optional
            Size of the rolling window of lookback , by default 8
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        """
        self.lookback_period = lookback_period
        self.min_periods = min_periods

    def _first_lb(self, x, first_len):
        return (np.max(x) - np.min(x)) / first_len

    def _second_lb(self, x, first_len):
        half = int((first_len) / 2)
        return (np.max(x.iloc[:half]) - np.min(x.iloc[:half])) / half

    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], first_fit: bool = True):
        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Moving features require past values for calculation.
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        """

        if first_fit:
            self._first_object = weighted_window_features()
            self._second_object = weighted_window_features()
            self._third_object = weighted_window_features()

        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()

        fama = pd.DataFrame(
            np.zeros(dataframe.shape), columns=dataframe.columns, index=dataframe.index
        )
        if first_fit:
            fama = pd.concat(
                [pd.DataFrame(np.zeros((1, fama.shape[1])), columns=fama.columns), fama]
            )
        else:
            fama = pd.concat([self.values_from_last_run, fama])
        fama["_iloc"] = np.arange(len(fama))
        ll = [x for x in fama.columns if x != "_iloc"]

        first_res = self._first_object._template_feature_calculation(
            function_name="_first_object",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=int((self.lookback_period) / 2),
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._first_lb,
            operation_args=(int((self.lookback_period) / 2)),
        )

        second_res = self._second_object._template_feature_calculation(
            function_name="_second_object",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._second_lb,
            operation_args=(self.lookback_period),
        )

        third_res = self._third_object._template_feature_calculation(
            function_name="_third_object",
            win_function=_identity_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=self.lookback_period,
            min_periods=self.min_periods,
            symmetric=None,
            operation=self._first_lb,
            operation_args=(self.lookback_period),
        )
        fractal_dimension = (
            np.log(second_res + first_res) - np.log(third_res)
        ) / np.log(2)
        a_value = np.exp(-4.6 * (fractal_dimension - 1)).fillna(0)
        a_value = pd.DataFrame(
            np.where(a_value < 0.01, 0.01, a_value), columns=a_value.columns
        )

        FC = (self.lookback_period) / 2
        SC = self.lookback_period
        oldN = (2 - a_value) / a_value
        newN = ((SC - FC) * (oldN - 1) / (SC - 1)) + FC
        a_value = 2 / (newN + 1)

        for r1, r2, r3 in zip(
            dataframe.iterrows(), fama[1:].iterrows(), a_value.iterrows()
        ):
            previous_kama = fama[fama["_iloc"] == (r2[1]["_iloc"] - 1)][ll]
            fama.loc[fama["_iloc"] == r2[1]["_iloc"], ll] = (
                np.multiply(previous_kama.values, (1 - r3[1].values))
                + np.multiply(r1[1].values, r3[1].values)
            )[0]

        res = fama.iloc[1:][ll]

        self.values_from_last_run = res.iloc[-1:]
        return res


class TripleExponentialMovingFeature:
    """
    Provided dataframe must be in ascending order.
    """

    def __init__(
        self,
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


    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], first_fit: bool = True):

        """
        For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test/subsequent implementation pass fit_first=False

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Moving features require past values for calculation.
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True

        """
        if first_fit:
            self._first_exponential_average_object = ExponentialMovingFeature(
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
                com=self.com,
                span=self.span,
                halflife=self.halflife,
                alpha=self.alpha,
                min_periods=self.min_periods,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
                operation=self.operation,
            )
            self._second_exponential_average_object = ExponentialMovingFeature(
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
                com=self.com,
                span=self.span,
                halflife=self.halflife,
                alpha=self.alpha,
                min_periods=self.min_periods,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
                operation=self.operation,
            )
            self._third_exponential_average_object = ExponentialMovingFeature(
                initialize_using_operation=self.initialize_using_operation,
                initialize_span=self.initialize_span,
                com=self.com,
                span=self.span,
                halflife=self.halflife,
                alpha=self.alpha,
                min_periods=self.min_periods,
                ignore_na=self.ignore_na,
                axis=self.axis,
                times=self.times,
                operation=self.operation,
            )

        first_exponential_average = self._first_exponential_average_object.fit(
            dataframe=dataframe, first_fit=first_fit
        )
        second_exponential_average = self._second_exponential_average_object.fit(
            dataframe=first_exponential_average, first_fit=first_fit
        )
        third_exponential_average = self._third_exponential_average_object.fit(
            dataframe=second_exponential_average, first_fit=first_fit
        )
        triple_exponential_average = (
            3 * first_exponential_average
            - 3 * second_exponential_average
            + third_exponential_average
        )
        return triple_exponential_average


class SmoothedMovingAverage:
    """
    Provided dataframe must be in ascending order.
    """

    def __init__(self,
            lookback_period: int = 4):
        """
        Parameters
        ----------
        lookback_period : int, optional
            Size of the rolling window of lookback , by default 4
        """
        self.lookback_period = lookback_period

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
            Moving features require past values for calculation.
            Use True, when calculating for training data (very first fit)
            Use False, when calculating for subsequent testing/production data { in which case the values, which
            were saved during the last phase, will be utilized for calculation }, by default True

        """

        if first_fit:
            self._first_object = weighted_window_features()
            

        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()

        sma = pd.DataFrame(
            np.zeros(dataframe.shape), columns=dataframe.columns, index=dataframe.index
        )
        if first_fit:
            sma.iloc[self.lookback_period - 1] = (
                dataframe.iloc[: (self.lookback_period)].sum() / self.lookback_period
            )
        else:
            sma = pd.concat([self.values_from_last_run, sma])

        sma["_iloc"] = np.arange(len(sma))
        ll = [x for x in sma.columns if x != "_iloc"]

        _start_frame = self.lookback_period if first_fit else 0
        _start_sma = self.lookback_period if first_fit else 1
        for r1, r2 in zip(
            dataframe[_start_frame:].iterrows(), sma[_start_sma:].iterrows()
        ):

            previous_kama = sma[sma["_iloc"] == (r2[1]["_iloc"] - 1)][ll]

            sma.loc[sma["_iloc"] == r2[1]["_iloc"], ll] = (
                (previous_kama * (self.lookback_period - 1) + r1[1])
                / self.lookback_period
            ).values[0]

        res = sma[ll] if first_fit else sma.iloc[1:][ll]

        self.values_from_last_run = res.iloc[-1:]
        return res
