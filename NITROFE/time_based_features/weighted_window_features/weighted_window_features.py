from pandas.core.series import Series
from weighted_windows import (
    _barthann_window,
    _bartlett_window,
    _equal_window,
    _blackman_window,
    _blackmanharris_window,
    _bohman_window,
    _cosine_window,
    _exponential_window,
    _flattop_window,
    _gaussian_window,
    _hamming_window,
    _hann_window,
    _kaiser_window,
    _parzen_window,
    _triang_window,
    _weighted_moving_window,
)

import numpy as np
import pandas as pd
from typing import Union, Callable


class weighted_window_features:
    def __init__(self):
        self.params = {}
        pass

    def first_fit_params_save(self, function_name, **kwargs):

        if not function_name in self.params:
            self.params[function_name] = {}

        for _key in kwargs.keys():
            self.params[function_name][_key] = kwargs[_key]

    def _template_feature_calculation(
        self,
        function_name,
        win_function,
        first_fit: bool = True,
        dataframe: Union[pd.DataFrame, pd.Series] = None,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = (),
        **kwargs
    ):
        _function_name = function_name

        if not isinstance(operation_args, tuple):
            operation_args = (operation_args,)

        if first_fit:
            self.first_fit_params_save(
                _function_name,
                window=eval("window"),
                min_periods=eval("min_periods"),
                symmetric=eval("symmetric"),
                operation=eval("operation"),
                operation_args=eval("operation_args"),
                kwargs=kwargs
            )
        
        if not first_fit:
            if self.params[_function_name]["last_values_from_previous_run"] is None:
                raise ValueError(
                    "First fit has not occured before. Kindly run first_fit=True for first fit instance,"
                    "and then proceed with first_fit=False for subsequent fits "
                )
            dataframe = pd.concat(
                [
                    self.params[_function_name]["last_values_from_previous_run"],
                    dataframe,
                ],
                axis=0,
            )

        _return = dataframe.rolling(
            window=self.params[_function_name]["window"],
            min_periods=self.params[_function_name]["min_periods"],
        ).agg(
            lambda x: self.params[_function_name]["operation"](
                win_function(
                    data=x,
                    window_size=self.params[_function_name]["window"],
                    symmetric=self.params[_function_name]["symmetric"],
                    **self.params[_function_name]["kwargs"]
                ),
                *self.params[_function_name]["operation_args"]
            )
        )
        if not first_fit:
            _return = _return.iloc[self.params[_function_name]["window"] - 1 :]
        self.first_fit_params_save(
            _function_name,
            last_values_from_previous_run=dataframe.iloc[
                1 - self.params[_function_name]["window"] :
            ],
        )

        return _return

    def caluclate_weighted_moving_window_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        operation: Callable = np.sum,
        operation_args: tuple = (),
    ):
        """
        Create weighted moving window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for self defined operation function

        Returns
        -------
            final value after operation

        References
        -----
        .. [1] Wikipedia, "Weighted Window function",
            https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

        """
        _function_name = "caluclate_weighted_moving_window_feature"
        return self._template_feature_calculation(
            function_name=_function_name,
            win_function=_weighted_moving_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=window,
            min_periods=min_periods,
            symmetric=None,
            operation=operation,
            operation_args=operation_args,
        )

    def caluclate_barthann_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = ()
    ):
        """
        Create Bartlett–Hann weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which Bartlett–Hann weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        Returns
        -------
            final value after operation

        References
        -----
        .. [1] Wikipedia, "Window function",
            https://en.wikipedia.org/wiki/Window_function#Bartlett%E2%80%93Hann_window

        """
        _function_name = "caluclate_barthann_feature"
        return self._template_feature_calculation(
            function_name=_function_name,
            win_function=_barthann_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=window,
            min_periods=min_periods,
            symmetric=symmetric,
            operation=operation,
            operation_args=operation_args,
        )

    def caluclate_bartlett_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = (),
    ):
        """
        Create bartlett weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which bartlett weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        Returns
        -------
            final value after operation

        References
        -----
        .. [1] Center for Computer Research in Music and Acoustics, "Bartlett (``Triangular'') Window",
            https://ccrma.stanford.edu/~jos/sasp/Bartlett_Triangular_Window.html
        """
        _function_name = "caluclate_bartlett_feature"
        return self._template_feature_calculation(
            function_name=_function_name,
            win_function=_bartlett_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=window,
            min_periods=min_periods,
            symmetric=symmetric,
            operation=operation,
            operation_args=operation_args,
        )

    def caluclate_equal_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        operation: Callable = np.mean,
        operation_args: tuple = (),
    ):
        """
        Create equally weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which equally weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        Returns
        -------
            final value after operation
        """
        _function_name = "caluclate_equal_feature"
        return self._template_feature_calculation(
            function_name=_function_name,
            win_function=_equal_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=window,
            min_periods=min_periods,
            symmetric=None,
            operation=operation,
            operation_args=operation_args,
        )

    def caluclate_blackman_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = (),
    ):
        """
        Create blackman weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which blackman weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function
        Returns
        -------
            final value after operation

        References
        -----
        .. [1] Wikipedia, "Window function",
            https://en.wikipedia.org/wiki/Window_function#Blackman_window
        """
        _function_name = "caluclate_blackman_feature"
        return self._template_feature_calculation(
            function_name=_function_name,
            win_function=_blackman_window,
            first_fit=first_fit,
            dataframe=dataframe,
            window=window,
            min_periods=min_periods,
            symmetric=symmetric,
            operation=operation,
            operation_args=operation_args,
        )


    def caluclate_blackmanharris_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = (),
    ):
        """
        Create blackman-harris weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which blackman-harris weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function
        Returns
        -------
            final value after operation

        References
        -----
        .. [1] Wikipedia, "Window function",
            https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window
        """
        _function_name = "caluclate_blackmanharris_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_blackmanharris_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args,
            )

    def caluclate_bohman_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = ()
    ):
        """
        Create bohman weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which bohman weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        References
        -----
        .. [1] mathworks, "bohmanwin",
            https://www.mathworks.com/help/signal/ref/bohmanwin.html

        """
        _function_name = "caluclate_bohman_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_bohman_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args,
            )

    def caluclate_cosine_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = ()
    ):
        """
        Create cosine weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which cosine weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        References
        -----
        .. [1] Wikipedia, "Window function",
            https://en.wikipedia.org/wiki/Window_function#Power-of-sine/cosine_windows

        """
        _function_name = "caluclate_cosine_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_cosine_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args,
            )

    def caluclate_exponential_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = (),
        center: float = None,
        tau: float = 1,
    ):
        """
        Create exponential weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which exponential weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function
        center : float , optional
            Parameter defining the center location of the window function.
            The default value if not given is center = (M-1) / 2. This parameter must take its default value for symmetric windows.
        tau : float , optional
            Parameter defining the decay. For center = 0 use tau = -(M-1) / ln(x) if x is the fraction of the window remaining at the end, by default 1

        References
        -----
        .. [1] Wikipedia, "Window function",
            https://en.wikipedia.org/wiki/Window_function#Exponential_or_Poisson_window

        """
        _function_name = "caluclate_exponential_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_exponential_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args,
                center=center,
                tau = tau
            )


    def caluclate_flattop_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = (),
    ):
        """
        Create flattop weighted rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which flattop weighted rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        References
        -----
        .. [1] mathworks, "flattopwin",
            https://in.mathworks.com/help/signal/ref/flattopwin.html

        """
        _function_name = "caluclate_flattop_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_flattop_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args,
            )

    def caluclate_gaussian_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = (),
        std: float = 1
    ):
        """
        Create flattop gaussian rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which flattop gaussian rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function
        std : float, optional
            The standard deviation, sigma.

        References
        -----
        .. [1] Wikipedia, "Window function",
            https://en.wikipedia.org/wiki/Window_function#Gaussian_window

        """
        _function_name = "caluclate_gaussian_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_gaussian_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args,
                std=std
            )


    def caluclate_hamming_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = ()
    ):
        """
        Create flattop hamming rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which flattop hamming rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        References
        -----
        .. [1] Wikipedia, "Window function",
            https://en.wikipedia.org/wiki/Window_function#Gaussian_window

        """
        _function_name = "caluclate_hamming_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_hamming_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args
            )

    def caluclate_hann_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = ()
    ):
        """
        Create flattop hann rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which flattop hann rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        References
        -----
        .. [1] Wikipedia, "Hann function",
            https://en.wikipedia.org/wiki/Hann_function

        """
        _function_name = "caluclate_hann_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_hann_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args
            )

    def caluclate_kaiser_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        beta: float = 7,
        operation: Callable = np.mean,
        operation_args: tuple = ()
    ):
        """
        Create flattop kaiser rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which flattop kaiser rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True  
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        beta : float, optional
            Shape parameter, determines trade-off between main-lobe width and side lobe level, by default 7
            As beta gets large, the window narrows.
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        References
        -----
        .. [1] Wikipedia, "Kaiser window",
            https://en.wikipedia.org/wiki/Kaiser_window
        """
        _function_name = "caluclate_kaiser_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_kaiser_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args,
                beta=beta
            )


    def caluclate_parzen_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = ()
    ):
        """
        Create flattop parzen rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which flattop parzen rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True  
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function
        References
        -----
        .. [1] Wikipedia, "Parzen function",
            https://en.wikipedia.org/wiki/Window_function#Parzen_window

        """
        _function_name = "caluclate_parzen_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_parzen_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args
            )


    def caluclate_triang_feature(
        self,
        dataframe: Union[pd.DataFrame, pd.Series],
        first_fit: bool = True,
        window: int = 3,
        min_periods: int = 1,
        symmetric: bool = False,
        operation: Callable = np.mean,
        operation_args: tuple = ()
    ):
        """
        Create flattop triang rolling window feature

        Parameters
        ----------
        dataframe : Union[pd.DataFrame,pd.Series]
            dataframe/series over which flattop triang rolling window feature is to be constructed
        first_fit : bool, optional
            Rolling features require past "window" number of values for calculation.
            Use True, when calculating for training data { in which case last "window" number of values will be saved }
            Use False, when calculating for testing/production data { in which case the, last "window" number of values, which
            are were saved during the last phase, will be utilized for calculation }, by default True  
        window : int, optional
            Size of the rolling window, by default 3
        min_periods : int, optional
            Minimum number of observations in window required to have a value, by default 1
        symmetric : bool, optional
            When True , generates a symmetric window, for use in filter design. When False,
            generates a periodic window, for use in spectral analysis, by default False
        operation : Callable, optional
            operation to perform over the weighted rolling window values, by default np.mean
        operation_args : tuple, optional
            additional agrument values to be sent for operation function

        References
        -----
        .. [1] Wikipedia, "triang function",
            https://en.wikipedia.org/wiki/Window_function#Parzen_window

        """
        _function_name = "caluclate_triang_feature"
        return self._template_feature_calculation(
                function_name=_function_name,
                win_function=_triang_window,
                first_fit=first_fit,
                dataframe=dataframe,
                window=window,
                min_periods=min_periods,
                symmetric=symmetric,
                operation=operation,
                operation_args=operation_args
            )

##################################################################################

# df = pd.DataFrame({"a": np.random.random(14), "b": np.random.random(14) * 10})
# print(df)
# from scipy import signal
# ss=False
# ww=4
# def perc1(data, pp):
#     return np.percentile(data, pp)

# ob = weighted_window_features()
# res_all = ob.caluclate_triang_feature(
#     dataframe=df,first_fit= True,window=ww, min_periods=2, operation=np.mean,symmetric=ss )


# #print(ob.params['caluclate_gaussian_feature']['last_values_from_previous_run'])


# res_comb = pd.concat(
#     [
#         ob.caluclate_triang_feature(
#             dataframe=df.iloc[:6],first_fit=True ,window=ww,min_periods= 2, operation=np.mean, symmetric=ss ),
#         ob.caluclate_triang_feature(dataframe=df.iloc[6:], first_fit=False),
#     ]
# )
# print(res_comb)

# print(pd.concat([res_all, res_comb], axis=1))

##################################################################################