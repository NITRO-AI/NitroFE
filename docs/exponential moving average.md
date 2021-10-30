

# Class ExponentialMovingFeature

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1QCmszvpWErCk9_u4rzeoEWRDgV63i0H0?usp=sharing)

## Import
`
from NitroFe import ExponentialMovingFeature
`

![ExponentialMovingFeature](https://media.giphy.com/media/kittGjKLRM6IZJSmRK/giphy.gif)

The exponential moving average is caluclated as

$$
\operatorname{ema[0]} = dataframe[0]
$$

$$
\operatorname{ema[t]} = (1-alpha)*ema[t-1] + alpha*x[t]
$$


if you want you calculate via the traditional way, in which the ema[0] isnt the first value in the dataframe
(usually a simple moving average over the first few values ), 

you can use the paramters 'initialize_using_operation' and 'initialize_span' ,
in which case the exponential moving avergae will be calculated as

$$
\operatorname{ema}[0 \to (initialize\_span - 2)] = Nan
$$

$$
\operatorname{ema}[initialize\_span - 1] = operation( dataframe[0 \to (initialize\_span-1) ] )
$$

$$
\operatorname{ema[t]} = (1-alpha)*ema[t-1] + alpha*x[t]
$$

## Methods

::: NitroFE.time_based_features.moving_average_features.moving_average_features.ExponentialMovingFeature
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
*  pydata, "ewm"

    [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html)