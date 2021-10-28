

# Class InverseFisherRelativeStrengthIndex

## Import
`
from NitroFe import InverseFisherRelativeStrengthIndex
`

![InverseFisherRelativeStrengthIndex](https://media.giphy.com/media/5NiDV7lIFuIzormSzv/giphy.gif)


Inverse Fisher Relative Strength Index is calculated as (IFRSI)

$$
RSI = Relative\_Strength\_Index
$$

$$
ranged\_RSI = 0.1 * (RSI - 50)
$$

$$
wRSI = weighted \ moving \ average \ of ranged\_RSI \ of \ size \ 'lookback\_for\_inverse\_fisher'
$$

$$
IFRSI = \frac{\exp(2*wRSI)-1}{\exp(2*wRSI)+1}
$$

For your training/initial fit phase (very first fit) use fit_first=True, and for any production/test implementation pass fit_first=False

## Methods

::: NitroFE.time_based_features.indicator_features._InverseFisherRelativeStrengthIndex.InverseFisherRelativeStrengthIndex
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* traders, "A Smoothed Rsi Inverse Fisher Transform",
    [http://traders.com/Documentation/FEEDbk_docs/2010/10/Vervoort.html](http://traders.com/Documentation/FEEDbk_docs/2010/10/Vervoort.html)