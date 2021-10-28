

# Class TypicalValue

## Import
`
from NitroFe import TypicalValue
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