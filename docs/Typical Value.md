

# Class TypicalValue

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1J4D6kt5Jh04YoEvzvrE2oRcN5dNaKKOm?usp=sharing)

## Import
`
from NitroFE import TypicalValue
`

![TypicalValue](https://media.giphy.com/media/kwABekFGGfQylsk50W/giphy.gif)

Typical value is calculated as

$$
Typical\_Value[t]= \frac{ \max{(dataframe[t-lookback\_period \to t])} +\min{(dataframe[t-lookback\_period \to t])} + dataframe[t] }{3}
$$


## Methods

::: NitroFE.time_based_features.indicator_features._TypicalValue.TypicalValue
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