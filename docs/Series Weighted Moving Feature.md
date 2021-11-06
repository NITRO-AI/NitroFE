
# Class SeriesWeightedMovingFeature

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1ifwYR9RhQw4WV5qM2CuhWWXAo-J8Oj-Q?usp=sharing)

## Import
`
from NitroFE import SeriesWeightedMovingFeature
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
