import pandas as pd
import numpy as np
from typing import Union, Callable

from NitroFE.time_based_features.weighted_window_features.weighted_window_features import (
    weighted_window_features,
)
from weighted_windows import _equal_window, _identity_window


class exponential_moving_feature:
    def __init__(self):
        self.last_values_from_previous_run = None

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
        exponential_moving_feature

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
        .. [1] pydata, "ewm",
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
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


class hull_moving_feature:
    def __init__(self):
        pass

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 4,
        min_periods: int = 1,
        operation: Callable = np.mean,
    ):
        """
        The Hull Moving Average (HMA), developed by Alan Hull, is an extremely fast and smooth moving average.
        In fact, the HMA almost eliminates lag altogether and manages to improve smoothing at the same time.

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean

        References
        -----
        .. [1] school.stockcharts, "Hull Moving Average (HMA)",
            https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average

        """
        if first_fit:
            self._window_size_weighted_moving_average_object = (
                weighted_window_features()
            )
            self._window_by_two_size_weighted_moving_average_object = (
                weighted_window_features()
            )
            self._hma_object = weighted_window_features()
            self.window = window
            self.min_periods = min_periods
            self.operation = operation

        if window <= 1:
            raise ValueError(f"window size less than equal to 1 not supported")
        self.window_by_two, self.window_square_root = int(
            np.ceil(self.window / 2)
        ), int(np.ceil(np.sqrt(self.window)))

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


class kaufman_adaptive_moving_average:
    def __init__(self):
        pass

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        kaufman_efficiency_lookback_period: int = 4,
        kaufman_efficiency_min_periods: int = 1,
        fast_ema_span: int = 2,
        slow_ema_span: int = 5,
    ):
        """
        kaufman_adaptive_moving_average

        Current value=  value in dataframe
        ER = Kaufman Efficiency Ratio
        fast ema alpha = 2/(1+fast_ema_span)
        slow ema alpha = 2/(1+slow_ema_span)

        Smoothing Constant (SC) =  (ER x ('fast ema alpha'- 'slow ema alpha' ) + 'slow ema alpha')**2

        Current kaufman adaptive moving average (KAMA )= Prior KAMA + SC x ( 'Current value'  - Prior KAMA)

        Parameters
        ----------
        dataframe : Union[pd.DataFrame, pd.Series]
            dataframe containing column values to create feature over
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            were saved during the last phase, will be utilized for calculation }, by default True
        kaufman_efficiency_lookback_period : int, optional
            Size of the rolling window of lookback for the caluculation of kaufman efficiency ratio , by default 4
        kaufman_efficiency_min_periods : int, optional
            Minimum number of observations in window required to have a value for kaufman efficiency ratio, by default 1
        fast_ema_span : int, optional
            fast span length, by default 2
        slow_ema_span : int, optional
            slow span length, by default 5

        References
        -----
        .. [1] school.stockcharts, "kaufman adaptive moving average",
            https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average

        """

        if first_fit:
            self._kaufman_object = _a_kaufman_efficiency()
            self.kaufman_efficiency_lookback_period = kaufman_efficiency_lookback_period
            self.kaufman_efficiency_min_periods = kaufman_efficiency_min_periods
            self.fast_ema_span = fast_ema_span
            self.slow_ema_span = slow_ema_span

        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()

        kma = pd.DataFrame(np.zeros(dataframe.shape), columns=dataframe.columns,index=dataframe.index)
        if first_fit:
            kma = pd.concat(
                [pd.DataFrame(np.zeros((1, kma.shape[1])), columns=kma.columns), kma]
            )
        else:
            kma = pd.concat([self.values_from_last_run, kma])
        kma["_iloc"] = np.arange(len(kma))

        _kaufman_efficiency = self._kaufman_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
            lookback_period=self.kaufman_efficiency_lookback_period,
            min_periods=self.kaufman_efficiency_min_periods,
        )

        ll = [x for x in kma.columns if x != "_iloc"]
        SC = (
            (
                _kaufman_efficiency
                * (2 / (self.fast_ema_span + 1) - 2 / (self.slow_ema_span + 1))
                + 2 / (self.slow_ema_span + 1)
            )** 2
        ).fillna(0)

        for r1, r2, r3 in zip(dataframe.iterrows(), kma[1:].iterrows(), SC.iterrows()):

            previous_kama = kma[kma["_iloc"] == (r2[1]["_iloc"] - 1)][ll]

            kma.loc[kma["_iloc"] == r2[1]["_iloc"], ll] = (
                previous_kama + np.multiply( r3[1].values,(r1[1].values - previous_kama.values) )
            ).values[0]

        res = kma.iloc[1:][ll]

        self.values_from_last_run = res.iloc[-1:]

        return res


class triple_exponential_moving_feature:
    def __init__(self):
        pass

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        com: float = None,
        span: float = None,
        halflife: float = None,
        alpha: float = None,
        min_periods: int = 0,
        ignore_na: bool = False,
        axis: int = 0,
        times: str = None,
        operation: str = "mean",
    ):

        """triple_exponential_moving_feature

        The triple exponential moving average (TEMA) was designed to smooth price fluctuations,
        thereby making it easier to identify trends without the lag associated with traditional moving averages (MA).
        It does this by taking multiple exponential moving averages (EMA) of the original EMA and subtracting out some of the lag.

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
            self._first_exponential_average_object = exponential_moving_feature()
            self._second_exponential_average_object = exponential_moving_feature()
            self._third_exponential_average_object = exponential_moving_feature()

            self.com = com
            self.span = span
            self.halflife = halflife
            self.alpha = alpha
            self.min_periods = min_periods
            self.ignore_na = ignore_na
            self.axis = axis
            self.times = times
            self.operation = operation

        first_exponential_average = self._first_exponential_average_object.fit(
            dataframe=dataframe,
            first_fit=first_fit,
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

        second_exponential_average = self._second_exponential_average_object.fit(
            dataframe=first_exponential_average,
            first_fit=first_fit,
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
        third_exponential_average = self._third_exponential_average_object.fit(
            dataframe=second_exponential_average,
            first_fit=first_fit,
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
        triple_exponential_average = (
            3 * first_exponential_average
            - 3 * second_exponential_average
            + third_exponential_average
        )
        return triple_exponential_average


#####################################################################

# ob = kaufman_adaptive_moving_average()
# df = pd.DataFrame(
#     {
#         "a": 20 * np.random.random(25) + np.random.choice(np.arange(-20, 20), size=25),
#         "b": np.arange(0, 100, step=2)[:25]
#         + np.random.choice(np.arange(-20, 20), size=25),
#     }
# )
# df = df["a"]
# wz = 6
# mp = 1
# flp = 10
# flp2 = 14
# print("df", df)

# res_all = ob.fit(df, first_fit=True).reset_index(drop=True)

# res_comb = pd.concat(
#     [
#         ob.fit(df.iloc[:flp], first_fit=True),
#         ob.fit(df.iloc[flp:flp2], first_fit=False),
#         ob.fit(df.iloc[flp2:], first_fit=False),
#     ],
#     axis=0,
# ).reset_index(drop=True)

# print(pd.concat([res_all, res_comb], axis=1))
# print((res_all.fillna(0) != res_comb.fillna(0)).sum())

#####################################################################
