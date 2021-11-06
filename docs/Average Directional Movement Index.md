
# Class AverageDirectionalMovementIndex

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/12vnEIF-QwKJrCHaksQkrkua5-hjlTOdL?usp=sharing)

## Import
`
from NitroFE import AverageDirectionalMovementIndex
`

![AverageDirectionalMovementIndex](https://media.giphy.com/media/grrI4t2AqnVf6YJaUy/giphy.gif)

The average directional index (ADX) is a technical analysis indicator used by some traders to determine the strength of a trend.

The trend can be either up or down, and this is shown by two accompanying indicators, the negative directional indicator (-DI) and the positive directional indicator (+DI). Therefore, the ADX commonly includes three separate lines. These are used to help assess whether a trade should be taken long or short, or if a trade should be taken at all.

The average directional index (ADX) is calculated as

$$
Raw\_Positive\_Directional\_Movement[t] =  \max{(dataframe[t-directional\_movement\_lookback\_period \to t])} -\max{(dataframe[t-2*directional\_movement\_lookback\_period \to t-directional\_movement\_lookback\_period])}
$$

$$
Raw\_Negative\_Directional\_Movement[t] =  \min{(dataframe[t-2*directional\_movement\_lookback\_period \to t-directional\_movement\_lookback\_period])} - \min{(dataframe[t-directional\_movement\_lookback\_period \to t)}
$$

$$
Positive\_Directional\_Movement[t] \ (+DM) =  if (Raw\_Positive\_Directional\_Movement[t] > Raw\_Negative\_Directional\_Movement[t]) \ and \ (Raw\_Positive\_Directional\_Movement[t] >0), then \ Raw\_Positive\_Directional\_Movement[t] \ else \ 0
$$

$$
Negative\_Directional\_Movement[t] \ (-DM) =  if (Raw\_Negative\_Directional\_Movement[t] > Raw\_Positive\_Directional\_Movement[t]) \ and \ (Raw\_Negative\_Directional\_Movement[t] >0), then \ Raw\_Negative\_Directional\_Movement[t] \ else \ 0
$$

$$
Positive\_Directional\_Index \ (+DI) =  100*\frac{Smoothed \ moving \ average \ of \ Positive\_Directional\_Movement \ with \ length \ 'directional\_movement\_smoothing\_period' }{Average \ True \ Range \ of \ dataframe}
$$

$$
Negative\_Directional\_Index \ (-DI) =   100*\frac{Smoothed \ moving \ average \ of \ Negative\_Directional\_Movement \ with \ length \ 'directional\_movement\_smoothing\_period' }{Average \ True \ Range \ of \ dataframe}
$$

$$
average \ directional \ index \ (ADX) = Smoothed \ moving \ average \ of \ 100*\frac{+DI - -DI}{+DI + -DI} \ with \ length \ 'average\_directional\_movement\_smoothing\_period'
$$
## Methods

::: NitroFE.time_based_features.indicator_features._AverageDirectionalMovementIndex.AverageDirectionalMovementIndex
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* wikipedia, "Average directional movement index",
    [https://en.wikipedia.org/wiki/Average_directional_movement_index](https://en.wikipedia.org/wiki/Average_directional_movement_index)
