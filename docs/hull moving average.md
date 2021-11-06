
# Class HullMovingFeature

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1LnKtew_FdhdivCWJSep_5H-3Aw5P-SxR?usp=sharing)


## Import
`
from NitroFE import HullMovingFeature
`

![HullMovingFeature](https://media.giphy.com/media/y9rdw8zH5dlYWBJkJI/giphy.gif)

The Hull Moving Average (HMA), developed by Alan Hull, is an extremely fast and smooth moving average.

In fact, the HMA almost eliminates lag altogether and manages to improve smoothing at the same time.

The Hull Moving Average is caluclated as

$$
\operatorname{WMA_(\frac{window}{2})} = Weighted \ Moving \ Average \ of \ length \ \frac{window}{2} \ over \ dataframe
$$

$$
\operatorname{WMA_(window)} = Weighted \ Moving \ Average \ of \ length \ window \ over \ dataframe
$$

$$
\operatorname{Raw \ Hull \ Moving \ Average} = 2*\operatorname{WMA_(\frac{window}{2})} - \operatorname{WMA_(window)}
$$

$$
\operatorname{Hull \ Moving \ Average} =  Weighted \ Moving \ Average \ of \ length \ \sqrt(window) over \operatorname{Raw \ Hull \ Moving \ Average}
$$

## Methods

::: NitroFE.time_based_features.moving_average_features.moving_average_features.HullMovingFeature
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
-----
* school.stockcharts, "Hull Moving Average (HMA)",
    [https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average](https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average)