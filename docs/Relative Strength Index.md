

# Class RelativeStrengthIndex

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1KBPprLc9lDPZTOXWsMh06JjANor15Yxp?usp=sharing)

## Import
`
from NitroFe import RelativeStrengthIndex
`

![RelativeStrengthIndex](https://media.giphy.com/media/588gu1EXpuqJxvl9mk/giphy.gif)

The RSI is classified as a momentum oscillator, measuring the velocity and magnitude of price movements. Momentum is the rate of the rise or fall in price. 

$$
Difference\_value[t] (DFV) = dataframe[t] - dataframe[t-1]
$$

$$
Upwards\_movement[t] (UM) = DFV[t] \ if \ DFV[t]>0 \ else \ 0
$$

$$
Downward\_movement[t] (DM) = |DFV[t]| \ if \ DFV[t]<0 \ else \ 0
$$

$$
RS = \frac{Smoothed \ moving \ average \ of \ UM \ over \ 'lookback\_period' \ period }{Smoothed \ moving \ average \ of \ DM \ over \ 'lookback\_period' \ period}
$$

$$
RSI = 100 -\frac{100}{1+RS}
$$



## Methods

::: NitroFE.time_based_features.indicator_features._RelativeStrengthIndex.RelativeStrengthIndex
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* wikipedia, "Relative strength index",
    [https://en.wikipedia.org/wiki/Relative_strength_index](https://en.wikipedia.org/wiki/Relative_strength_index)