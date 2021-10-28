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


class ElasticSeriesWeightedAverage:
    def __init__(self,
        weight_sum_lookback: int = 4):
        """
        Parameters
        ----------
        weight_sum_lookback : int, optional
            Size of the rolling window of lookback , by default 4
        """
        self.weight_sum_lookback = weight_sum_lookback

    def fit(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        dataframe_for_weight: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True):
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