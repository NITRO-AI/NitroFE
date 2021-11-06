

# Class BollingerBands

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1OL27jZJ5-6YdPrpu5Nl61AFi1L6-Xm-W?usp=sharing)

## Import
`
from NitroFE import BollingerBands
`

![BollingerBands](https://media.giphy.com/media/dS8KB6cIGLa0sLr2bN/giphy.gif)

Bollinger Bands are a type of statistical chart characterizing the prices and volatility over time of a financial instrument or commodity, using a formulaic method propounded by John Bollinger in the 1980s. 

Bollinger Bands are caluclated as 

$$
Typical\_Value[t] (TV)= \frac{ \max{(dataframe[t-lookback\_period \to t])} +\min{(dataframe[t-lookback\_period \to t])} + dataframe[t] }{3}
$$

$$
MA \ of \ typical \ value \ (MATV) = Moving \ average \ of \ typical \ value \ with \ 'moving_average_typical_value_lookback_period' \ window \ size
$$

$$
STD \ of \ typical \ value \ (STD TV)= Standard \ deviation \ of \ typical \ value \ with \ 'moving_average_typical_value_lookback_period' \ window \ size
$$

$$
Positive \ band = (MATV) + 'standard\_deviation\_multiplier' * (STD TV)
$$

$$
Negative band = (MATV) - 'standard\_deviation\_multiplier' * (STD TV)
$$


## Methods

::: NitroFE.time_based_features.indicator_features._BollingerBands.BollingerBands
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* wikipedia, "Bollinger_Bands",
    [https://en.wikipedia.org/wiki/Bollinger_Bands](https://en.wikipedia.org/wiki/Bollinger_Bands)