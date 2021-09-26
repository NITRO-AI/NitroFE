import numpy as np
import pandas as pd
from typing import Union, Callable


def calculate_moving_average_convergence_divergence(dataframe: Union[pd.DataFrame, pd.Series],
                                                    min_periods: int = 1,
                                                    fast_period: int = 12,
                                                    slow_period: int = 26,
                                                    smoothing_period: int = 9,
                                                    operation: str = 'mean',
                                                    use_ema_for_smoothing: bool = True):
    """calculate_moving_average_convergence_divergence 

    [extended_summary]

    Parameters
    ----------
    dataframe : Union[pd.DataFrame, pd.Series]
        [description]
    window : int, optional
        [description], by default 3
    min_periods : int, optional
        [description], by default 1
    fast_period : int, optional
        [description], by default 12
    slow_period : int, optional
        [description], by default 26
    smoothing_period : int, optional
        [description], by default 9
    use_ema_for_smoothing : bool, optional
        [description], by default True
    """
    fast_ema = dataframe.ewm(fast_period, min_periods).mean()
