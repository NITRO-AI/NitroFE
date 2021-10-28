
# Class HullMovingFeature

![hull_moving_feature](https://media.giphy.com/media/y9rdw8zH5dlYWBJkJI/giphy.gif)

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