from weighted_window_features import (
    caluclate_equal_feature,
    caluclate_barthann_feature,
    caluclate_bartlett_feature,
    caluclate_blackman_feature,
    caluclate_blackmanharris_feature,
    caluclate_bohman_feature,
    caluclate_cosine_feature,
    caluclate_exponential_feature,
    caluclate_flattop_feature,
    caluclate_gaussian_feature,
    caluclate_hamming_feature,
    caluclate_hann_feature,
    caluclate_kaiser_feature,
    caluclate_parzen_feature,
    caluclate_triang_feature,
)


import numpy as np
import pandas as pd
from typing import Union


class weighted_rolling_window_engine:
    def __init__(self):
        self.function_mapper = {
            "equal": caluclate_equal_feature,
            "barthann": caluclate_barthann_feature,
            "bartlett": caluclate_bartlett_feature,
            "blackman": caluclate_blackman_feature,
            "blackmanharris": caluclate_blackmanharris_feature,
            "bohman": caluclate_bohman_feature,
            "cosine": caluclate_cosine_feature,
            "exponential": caluclate_exponential_feature,
            "flattop": caluclate_flattop_feature,
            "gaussian": caluclate_gaussian_feature,
            "hamming": caluclate_hamming_feature,
            "hann": caluclate_hann_feature,
            "kaiser": caluclate_kaiser_feature,
            "parzen": caluclate_parzen_feature,
            "triang": caluclate_triang_feature,
        }

    def fit(self, dataframe: Union[pd.DataFrame, pd.Series], payload: dict):
        """
        Parameters
        ----------
        dataframe :  Union[pd.DataFrame,pd.Series]
            dataframe/series over which barthann weighted rolling window feature is to be constructed
        payload : dict
            payload containing feature generation information

        Returns
        -------
        dict

        """
        self.output_dict = {}
        self.dataframe = dataframe
        self.payload = payload

        for _column_key in self.payload.keys():

            if "weighted_window_features" in self.payload[_column_key].keys():
                self.output_dict[_column_key] = {"weighted_window_features": {}}

                for _window_keys in self.payload[_column_key][
                    "weighted_window_features"
                ].keys():
                    if _window_keys in self.function_mapper.keys():

                        self.output_dict[_column_key]["weighted_window_features"][
                            _window_keys
                        ] = {}
                        dict_all = self.payload[_column_key][
                            "weighted_window_features"
                        ][_window_keys]

                        feature_generated = []
                        if isinstance(
                            self.payload[_column_key]["weighted_window_features"][
                                _window_keys
                            ]["window"],
                            int,
                        ):
                            iter_duration = 1
                            resulting_feature = self.function_mapper[_window_keys](
                                self.dataframe[_column_key],
                                **self.payload[_column_key]["weighted_window_features"][
                                    _window_keys
                                ]
                            )
                            feature_generated.append(resulting_feature)

                        else:
                            iter_duration = len(
                                self.payload[_column_key]["weighted_window_features"][
                                    _window_keys
                                ]["window"]
                            )
                            for _iter in range(iter_duration):
                                resulting_feature = self.function_mapper[_window_keys](
                                    self.dataframe[_column_key],
                                    **dict(
                                        zip(
                                            dict_all.keys(),
                                            np.array(list(dict_all.values()))[:, _iter],
                                        )
                                    )
                                )
                                feature_generated.append(resulting_feature)

                        self.output_dict[_column_key]["weighted_window_features"][
                            _window_keys
                        ] = feature_generated
        return self.output_dict


df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 2})
payload = {
    "a": {
        "weighted_window_features": {
            "barthann": {
                "window": [3, 4],
                "min_periods": [1, 2],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "bartlett": {
                "window": [3, 4],
                "min_periods": [1, 2],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "equal": {
                "window": 3,
                "min_periods": 1,
                "symmetric": False,
                "operation": np.mean,
            },
            "blackman": {
                "window": [3, 3],
                "min_periods": [1, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "blackmanharris": {
                "window": [3, 3],
                "min_periods": [1, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "bohman": {
                "window": [3, 3],
                "min_periods": [1, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "cosine": {
                "window": [3, 3],
                "min_periods": [1, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "exponential": {
                "window": [3, 5],
                "min_periods": [1, 1],
                "center": [1, None],
                "symmetric": [False, True],
                "tau": [1, 0.5],
                "operation": [np.mean, np.mean],
            },
            "flattop": {
                "window": [3, 6],
                "min_periods": [2, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.sum],
            },
            "gaussian": {
                "window": [3, 3],
                "min_periods": [1, 1],
                "symmetric": [False, True],
                "std": [1, 2],
                "operation": [np.mean, np.mean],
            },
            "hamming": {
                "window": [3, 3],
                "min_periods": [2, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "hann": {
                "window": [3, 3],
                "min_periods": [2, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "kaiser": {
                "window": [3, 3],
                "min_periods": [2, 1],
                "symmetric": [False, True],
                "beta": [7, 11],
                "operation": [np.mean, np.mean],
            },
            "parzen": {
                "window": [3, 3],
                "min_periods": [2, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
            "triang": {
                "window": [3, 3],
                "min_periods": [2, 1],
                "symmetric": [False, True],
                "operation": [np.mean, np.mean],
            },
        }
    }
}
temp = weighted_rolling_window_engine()
result_dict = temp.fit(df, payload)
print(result_dict["a"]["weighted_window_features"]["triang"])
