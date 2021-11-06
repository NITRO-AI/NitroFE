
# Class KeltnerChannel

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1SVTmxDEkESLxY5CqWNgrDBckMWu1X0RB?usp=sharing)

## Import
`
from NitroFE import KeltnerChannel
`

![KeltnerChannel](https://media.giphy.com/media/liXYyVogt1wmwkkSqd/giphy.gif)

keltner channel (KC) is calculated as 

$$
ATR = average \ true \ range \ of \ dataframe
$$

$$
EMA = Exponential \ moving \ average \ of \ dataframe \ over \ 'ema\_span' \ span
$$

$$
KC \ upperband = EMA + atr\_multiply * ATR
$$

$$
KC \ lowerband = EMA - atr\_multiply * ATR
$$



## Methods

::: NitroFE.time_based_features.indicator_features._KeltnerChannel.KeltnerChannel
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit
