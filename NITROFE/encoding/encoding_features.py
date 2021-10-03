from typing import Union, Callable, List
import pandas as pd
import numpy as np


class base_encoding:
    def __init__(
        self, columns_to_encode: Union[None, int, str, List[Union[str, int]]] = None
    ):
        if columns_to_encode:
            self.columns_to_encode = self._check_columns_to_encode(columns_to_encode)
        else:
            self.columns_to_encode = None

    def _check_columns_to_encode(self, columns_to_encode):
        if isinstance(columns_to_encode, list):
            if not all(isinstance(i, (str, int)) for i in columns_to_encode):
                raise ValueError(
                    "columns_to_encode should be a string, an int or a list of strings or integers."
                )
        else:
            if not isinstance(columns_to_encode, (str, int)):
                raise ValueError(
                    "columns_to_encode should be a string, an int or a list of strings or integers."
                )
            else:
                columns_to_encode = [columns_to_encode]
        return columns_to_encode

    def _check_fit_columns_to_encode(self, columns_to_encode, dataframe):
        if columns_to_encode:
            self.columns_to_encode = self._check_columns_to_encode(columns_to_encode)

        elif self.columns_to_encode is None:
            self.columns_to_encode = list(
                dataframe.select_dtypes(include=["O", "category"]).columns
            )
            if len(self.columns_to_encode) == 0:
                raise ValueError(
                    "No categorical variables in this dataframe. Please check the "
                    "variables format with pandas dtypes"
                )

    def _check_operations(self, operations):
        if not isinstance(operations, list):
            operations = [operations]
        self.operations = operations

    def _handle_concatenated_dataframe_column_names(self, y, dataframe):
        if not y.empty:
            if isinstance(y, pd.DataFrame):
                y_cols = list(y.columns)
            elif isinstance(y, pd.Series):
                y_cols = [y.name]
        else:
            y_cols = []
        self.concatenated_dataframe = pd.concat(
            [dataframe[[x for x in dataframe.columns if x not in y_cols]], y], axis=1
        )

        self.concatenated_dataframe.columns = [
            x for x in dataframe.columns if x not in y_cols
        ] + y_cols
        self.target_columns = y_cols

    def _check_weight_of_overall(self):
        if isinstance(self.weight_of_overall, float):
            if (self.weight_of_overall >= 0) & (self.weight_of_overall <= 1):
                self.weight_of_overall = (
                    self.weight_of_overall * self.concatenated_dataframe.shape[0]
                )
            else:
                raise ValueError(
                    "weight_of_overall if provided in float, cannot be greater than 1, less than 0"
                )
        elif not isinstance(self.weight_of_overall, int):
            raise ValueError("weight_of_overall only accepts int/float values")


class categorical_encoding(base_encoding):
    def __init__(
        self, columns_to_encode: Union[None, int, str, List[Union[str, int]]] = None
    ):
        super().__init__(columns_to_encode)

    def fit(
        self,
        dataframe: pd.DataFrame,
        y: Union[None, pd.Series, pd.DataFrame] = None,
        columns_to_encode: Union[None, int, str, List[Union[str, int]]] = None,
        operations: Union[None, List[Callable]] = [np.mean],
        payload: dict = None,
    ):

        self.encoding_dict = {}
        self.payload = payload
        if y is None:
            y = pd.DataFrame()

        self._handle_concatenated_dataframe_column_names(y, dataframe)

        if self.payload is None:
            self._check_fit_columns_to_encode(columns_to_encode, dataframe)
            self._check_operations(operations)

            self.payload = {
                _col: {
                    target_items: self.operations
                    for target_items in self.target_columns
                }
                for _col in self.columns_to_encode
            }

        for _col in self.payload.keys():

            _col_frame = self.concatenated_dataframe.groupby([_col]).agg(
                self.payload[_col]
            )
            _col_frame.columns = [
                _col + "_groupby_" + "target_" + lvlzero + "_"
                for lvlzero in _col_frame.columns.get_level_values(0)
            ] + _col_frame.columns.get_level_values(1)
            self.encoding_dict[_col] = _col_frame

        return self.encoding_dict


class smoothed_encoding(base_encoding):
    def __init__(
        self, columns_to_encode: Union[None, int, str, List[Union[str, int]]] = None
    ):
        super().__init__(columns_to_encode)

    def fit(
        self,
        dataframe: pd.DataFrame,
        y: Union[None, pd.Series, pd.DataFrame] = None,
        columns_to_encode: Union[None, int, str, List[Union[str, int]]] = None,
        operations: Union[None, List[Callable]] = [np.mean],
        weight_of_overall: Union[float, int] = 0.3,
        payload: dict = None,
    ):

        self.encoding_dict = {}
        self.payload = payload
        self.weight_of_overall = weight_of_overall
        if y is None:
            y = pd.DataFrame()

        self._handle_concatenated_dataframe_column_names(y, dataframe)

        if self.payload is None:
            self._check_fit_columns_to_encode(columns_to_encode, dataframe)
            self._check_operations(operations)

            self.payload = {
                _col: {
                    target_items: self.operations
                    for target_items in self.target_columns
                }
                for _col in self.columns_to_encode
            }

        self._check_weight_of_overall()

        for _col in self.payload.keys():

            _col_frame = (
                self.concatenated_dataframe.groupby([_col])
                .agg(self.payload[_col])
                .multiply(
                    self.concatenated_dataframe.groupby([_col])[_col].count(), axis=0
                )
                + self.weight_of_overall
                * np.concatenate(
                    [
                        self.concatenated_dataframe.agg(
                            {x: self.payload[_col][x]}
                        ).values.flatten()
                        for x in payload[_col].keys()
                    ]
                ).ravel()
            ).div(
                self.weight_of_overall
                + self.concatenated_dataframe.groupby([_col])[_col].count(),
                axis=0,
            )
            _col_frame.columns = [
                _col + "_groupby_" + "target_smoothed_" + lvlzero + "_"
                for lvlzero in _col_frame.columns.get_level_values(0)
            ] + _col_frame.columns.get_level_values(1)
            self.encoding_dict[_col] = _col_frame

        return self.encoding_dict


df = pd.DataFrame(
    {
        "a": 10 * np.random.random(10),
        "z": 20 * np.random.random(10),
        "b": np.random.choice(["ab", "cd"], size=10),
        "c": np.random.choice([0.1, 1.1], size=10),
        "d": np.random.choice(["ef", "gh"], size=10),
    }
)


ob = categorical_encoding()
rr = ob.fit(
    df,
    payload={
        "b": {"a": [np.sum, np.mean], "c": [np.median], "z": [np.std]},
        "d": {"a": [np.sum, np.mean], "c": [np.median], "z": [np.mean]},
    },
)
print(rr)

ob = smoothed_encoding()
rr = ob.fit(
    df,
    payload={
        "b": {"a": [np.sum, np.mean], "c": [np.median], "z": [np.std]},
        "d": {"a": [np.sum, np.mean], "c": [np.median], "z": [np.mean]},
    },
)
print(rr)
