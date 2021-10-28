
# Class KeltnerChannel

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
