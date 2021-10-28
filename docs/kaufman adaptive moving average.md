

# Class KaufmanAdaptiveMovingAverage

## Import
`
from NitroFe import KaufmanAdaptiveMovingAverage
`

![KaufmanAdaptiveMovingAverage](https://media.giphy.com/media/la1LKPepMbyCWd6HVv/giphy.gif)

Kaufman's Adaptive Moving Average (KAMA) is a moving average designed to account for noise or volatility.

KAMA will closely follow values when the values swings are relatively small and the noise is low.

KAMA will adjust when the values swings widen and follow values from a greater distance.

This trend-following indicator can be used to identify the overall trend, time turning points and filter price movements.

Kaufman adaptive moving average (KAMA) is calculated as :

$$
Current\_value[t] =  dataframe[t]
$$

$$
ER[t] = Kaufman\_Efficiency\_Ratio[t]
$$

$$
fast \ ema \ alpha = \frac{2}{1+fast\_ema\_span}
$$

$$
slow \ ema \ alpha = \frac{2}{1+slow\_ema\_span}
$$

$$
Smoothing\_Constant[t] = (ER[t] * (fast\_ema\_alpha - slow\_ema\_alpha ) + slow\_ema\_alpha)^{2}
$$

$$
KAMA[t]= KAMA[t-1] + Smoothing\_Constant[t] * ( Current\_value[t]  - KAMA[t-1] )
$$

## Methods

::: NitroFE.time_based_features.moving_average_features.moving_average_features.KaufmanAdaptiveMovingAverage
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit


References
-----
* school.stockcharts, "kaufman adaptive moving average",
    [https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average](https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average)
