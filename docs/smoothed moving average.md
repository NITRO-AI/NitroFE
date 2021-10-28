

# Class SmoothedMovingAverage

![smoothed_moving_average](https://media.giphy.com/media/LqMwgXOw1hMXgi5hGs/giphy.gif)

The Smoothed Moving Average (SMMA) is a combination of a SMA and an EMA.
It gives the recent values an equal weighting as the historic prices as it takes all available price data into account.
The main advantage of a smoothed moving average is that it removes short-term fluctuations,
and allows us to view the values trends much easier.

smoothed moving average (SMMA) is calculated as

$$
SMMA[t] = \frac{(SMMA[t-1]*(lookback\_period-1) + dataframe[t] )}{lookback\_period}
$$

## Methods

::: NitroFE.time_based_features.moving_average_features.moving_average_features.SmoothedMovingAverage
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit



References
------
* chartmill, "The SMOOTHED MOVING AVERAGE",
    [https://www.chartmill.com/documentation/technical-analysis-indicators/217-MOVING-AVERAGES-%7C-The-Smoothed-Moving-Average-%28SMMA%29](https://www.chartmill.com/documentation/technical-analysis-indicators/217-MOVING-AVERAGES-%7C-The-Smoothed-Moving-Average-%28SMMA%29)