
# Class SeriesWeightedMovingFeature

## Import
`
from NitroFe import SeriesWeightedMovingFeature
`

![SeriesWeightedMovingFeature](https://media.giphy.com/media/rfnnNjPxRoOVfPy5p7/giphy.gif)


Series Weighted Moving Feature  is a moving window variation and is calculated as 

$$
Series\_Weighted\_Moving\_Feature[t] = \frac{\operatorname{operation} (dataframe[t-'lookback\_period' \to t] * dataframe\_for\_weight[t-'lookback\_period' \to t])}{\operatorname{operation} (dataframe\_for\_weight[t-'lookback\_period' \to t])}
$$


## Methods

::: NitroFE.time_based_features.indicator_features._SeriesWeightedMovingFeature.SeriesWeightedMovingFeature
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit
