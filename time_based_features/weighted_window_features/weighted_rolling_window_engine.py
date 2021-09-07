from weighted_window_features import caluclate_barthann_feature, caluclate_bartlett_feature, caluclate_equal_feature, caluclate_blackman_feature
import numpy as np
import pandas as pd


class weighted_rolling_window_engine():
    """[summary]
    """

    def __init__(self):
        self.function_mapper = {'equal': caluclate_equal_feature,
                                'barthann': caluclate_barthann_feature,
                                'bartlett': caluclate_bartlett_feature,
                                'blackman':caluclate_blackman_feature}

    def fit(self, dataframe, payload):
        """[summary]

        Parameters
        ----------
        dataframe : [type]
            [description]
        payload : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        self.output_dict = {}
        self.dataframe = dataframe
        self.payload = payload

        for _column_key in self.payload.keys():

            if 'weighted_window_features' in self.payload[_column_key].keys():
                self.output_dict[_column_key] = {
                    'weighted_window_features': {}}

                for _window_keys in self.payload[_column_key]['weighted_window_features'].keys():
                    if (_window_keys in self.function_mapper.keys()):
                        self.output_dict[_column_key]['weighted_window_features'][_window_keys] = {
                        }
                        dict_dataframe = pd.DataFrame(self.payload[_column_key]['weighted_window_features'][_window_keys])
                        feature_generated = []
                        for _iter in range(len(dict_dataframe)):
                            resulting_feature = self.function_mapper[_window_keys](
                                self.dataframe[_column_key], **dict_dataframe.loc[_iter].to_dict())
                            feature_generated.append(resulting_feature)
                        self.output_dict[_column_key]['weighted_window_features'][_window_keys] = feature_generated
        return self.output_dict


df = pd.DataFrame({'a': np.arange(10), 'b': np.arange(10)*2})
payload = {'a': {'weighted_window_features':
                 {
                    'barthann': {'window': [3, 4],
                                  'min_periods': [1, 2],
                                  'symmetric': [False, True],
                                  'operation': [np.mean, np.sum]
                                  },
                    'bartlett': {'window': [3, 4],
                                  'min_periods': [1, 2],
                                  'symmetric': [False, True],
                                  'operation': [np.mean, np.sum]
                                  },
                    'equal':    {'window': 3,
                                  'min_periods': 1,
                                  'symmetric': False,
                                  'operation': [np.mean]
                                  },
                    'blackman': {'window': [3,6],
                                  'min_periods': [2,1],
                                  'symmetric': [False, True],
                                  'operation': [np.mean, np.sum]
                                  }
                 }
                 }
           }
temp = weighted_rolling_window_engine()
result_dict = temp.fit(df, payload)
print(result_dict)
