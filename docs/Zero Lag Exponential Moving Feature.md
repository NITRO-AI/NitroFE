

# Class ZeroLagExponentialMovingFeature

![ZeroLagExponentialMovingFeature](https://media.giphy.com/media/lW7PIM2hbP0fMw0roa/giphy.gif)


The zero lag exponential moving average (ZLEMA) indicator was created by John Ehlers and Ric Way

As is the case with the double exponential moving average (DEMA) and the triple exponential moving average (TEMA) and as indicated by the name, the aim is to eliminate the inherent lag associated to all trend following indicators which average a price over time.

$$
Lag = \frac{ 'lag_period' - 1 }{2}
$$

$$
EmaData[t] = dataframe[t] + (dataframe[t] - dataframe[t-'Lag'])
$$

$$
ZLEMA = Exponential \ moving \ average \ of \ 'EMAData'
$$


## Methods

::: NitroFE.time_based_features.indicator_features._ZeroLagExponentialMovingFeature.ZeroLagExponentialMovingFeature
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* wikipedia, "Zero lag exponential moving average",
    [https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average](https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average)