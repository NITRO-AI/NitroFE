


# Class KaufmanEfficiency

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