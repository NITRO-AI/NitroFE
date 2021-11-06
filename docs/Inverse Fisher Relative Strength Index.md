

# Class InverseFisherRelativeStrengthIndex

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1cY0_jDTlzk86pv1OLYehuXpZXIG-FMVo?usp=sharing)

## Import
`
from NitroFE import InverseFisherRelativeStrengthIndex
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