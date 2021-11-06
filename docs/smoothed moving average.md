

# Class SmoothedMovingAverage

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1mdA639urFTbtybZP7oR_0XHOLOYiiA0h?usp=sharing)

## Import
`
from NitroFE import SmoothedMovingAverage
`

![SmoothedMovingAverage](https://media.giphy.com/media/LqMwgXOw1hMXgi5hGs/giphy.gif)

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