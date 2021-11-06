


# Class KaufmanEfficiency

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1MHRQWl204rN-3GA-pBci4doMB0eacFdL?usp=sharing)

## Import
`
from NitroFE import KaufmanEfficiency
`

![KaufmanEfficiency](https://media.giphy.com/media/dlaAsgQ74zJA3U3ARb/giphy.gif)

It is calculated by dividing the net change in price movement over 'lookback_period' periods,
by the sum of the absolute net changes over the same 'lookback_period' periods.

For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

$$
Kaufman\_Efficiency[t]=  \frac{|dataframe[t]-dataframe[t-lookback\_period]|}{\sum_{i=t-lookback\_period+1}^{t}|dataframe[i]-dataframe[i-1]|}
$$


## Methods

::: NitroFE.time_based_features.indicator_features._KaufmanEfficiency.KaufmanEfficiency
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* strategyquant, "Kaufman’s Efficiency Ratio",
    [https://strategyquant.com/codebase/kaufmans-efficiency-ratio-ker/](https://strategyquant.com/codebase/kaufmans-efficiency-ratio-ker/)