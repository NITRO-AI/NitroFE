

# Class ZeroLagExponentialMovingFeature

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1iJ_ylVxPVpNRd0UOzcNq7woACMGcf9bV?usp=sharing)

## Import
`
from NitroFE import ZeroLagExponentialMovingFeature
`

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