
# Class AverageTrueRange

![AverageTrueRange](https://media.giphy.com/media/AQpueYRD0K39d5veh7/giphy.gif)


Average true range is a  volatility indicator.The indicator does not provide an indication of price trend,
simply the degree of volatility.The average true range is an N-period smoothed moving average of the true range values.

$$
first\_range[t]=  \max{(dataframe[t-true\_range\_lookback \to t]} -\min{dataframe[t-true\_range\_lookback \to t])}
$$

$$
second\_range[t]=  |{\max{(dataframe[t-true\_range\_lookback \to t]} - dataframe[t] )}|
$$

$$
third\_range[t]=  |{\min{(dataframe[t-true\_range\_lookback \to t]} - dataframe[t] )}|
$$

$$
True\_Range[t] = \max{ (first\_range[t], second\_range[t], third\_range[t]) }
$$

$$
Average\_True\_range[t] = \operatorname{Mean}(True\_Range[t-average\_true\_range\_span \to t])
$$


## Methods

::: NitroFE.time_based_features.indicator_features._AverageTrueRange.AverageTrueRange
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* investopedia, "Average True Range (ATR)",
    [https://www.investopedia.com/terms/a/atr.asp](https://www.investopedia.com/terms/a/atr.asp)