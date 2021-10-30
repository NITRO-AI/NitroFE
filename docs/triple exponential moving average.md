# Class TripleExponentialMovingFeature

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1f05_xV_ovDzm2urqbwzUAscaBmwHorT-?usp=sharing)

## Import
`
from NitroFe import TripleExponentialMovingFeature
`

![TripleExponentialMovingFeature](https://media.giphy.com/media/tyuKYKt8qe2dbP1Fqn/giphy.gif)

The Triple exponential moving average (TEMA) was designed to smooth value fluctuations,
thereby making it easier to identify trends without the lag associated with traditional moving averages (MA).
It does this by taking multiple exponential moving averages (EMA) of the original EMA and subtracting out some of the lag.

TEMA is calculated as

$$
EMA_1 = Exponential \ moving \ average \ of \ dataframe
$$

$$
EMA_2 = Exponential \ moving \ average \ of \ EMA_1
$$

$$
EMA_3 = Exponential \ moving \ average \ of \ EMA_2
$$

$$
TEMA = 3*EMA_1 - 3*EMA_2 + EMA_3
$$

## Methods

::: NitroFE.time_based_features.moving_average_features.moving_average_features.TripleExponentialMovingFeature
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit


References
-----
* investopedia, "triple-exponential-moving-average",
    [https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp](https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp)