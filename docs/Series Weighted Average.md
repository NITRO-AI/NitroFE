
# Class SeriesWeightedAverage

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1ifwYR9RhQw4WV5qM2CuhWWXAo-J8Oj-Q?usp=sharing)

## Import
`
from NitroFe import SeriesWeightedAverage
`

![SeriesWeightedAverage](https://media.giphy.com/media/4bgFEo3yfEzHX7LyqB/giphy.gif)

series_weighted_average
This feature is an implementation of volume weighted moving average

$$
series\_weighted\_average =  \frac{\sum (dataframe * dataframe\_for\_weight) }{\sum dataframe\_for\_weight }
$$


## Methods

::: NitroFE.time_based_features.indicator_features._SeriesWeightedAverage.SeriesWeightedAverage
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* investopedia, "Volume-Weighted Average Price",
    [https://www.investopedia.com/terms/v/vwap.asp](https://www.investopedia.com/terms/v/vwap.asp)