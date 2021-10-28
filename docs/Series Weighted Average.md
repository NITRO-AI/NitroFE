
# Class SeriesWeightedAverage

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