

# Class FractalAdaptiveMovingAverage

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1nDF0Th8iUyRsCFcrK51LZ9I4O93GlDfg?usp=sharing)

## Import
`
from NitroFE import FractalAdaptiveMovingAverage
`

![FractalAdaptiveMovingAverage](https://media.giphy.com/media/aWukNCTgI8nB1TagAC/giphy.gif)

Fractal Adaptive Moving Average Technical Indicator (FRAMA) was developed by John Ehlers.
This indicator is constructed based on the algorithm of the Exponential Moving Average,
in which the smoothing factor is calculated based on the current fractal dimension of the value series.
The advantage of FRAMA is the possibility to follow strong trend movements and to sufficiently slow down at the moments of price consolidation.

Fractal adaptive moving average (FAMA) is calculated as :


$$
N(t-Length,t) = \frac{\max(dataframe[t-Length \to t]) - \min(dataframe[t-Length \to t]) }{(t)-(t-Length)}
$$

$$
N1 = N(t-Length/2,t)
$$

$$
N2 = N(t-Length,t-Length/2)
$$

$$
N3 = N(t-Length,t)
$$

$$
D[t] = (\log(N1 + N2) - \log(N3))/\log(2)
$$

$$
A[t] = \exp(-4.6 * (D[t] - 1))
$$

$$
FAMA[t] = A[t] * dataframe[t] + (1 - A[t]) * FAMA[t-1]
$$

## Methods

::: NitroFE.time_based_features.moving_average_features.moving_average_features.FractalAdaptiveMovingAverage
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit


References
-----
* metatrader5, "Fractal Adaptive Moving Average",
    [https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/fama](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/fama)