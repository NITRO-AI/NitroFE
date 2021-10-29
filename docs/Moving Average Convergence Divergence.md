

# Class MovingAverageConvergenceDivergence

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1Haj2YNNp36emN47ok9J7fhXGiQUgYIac?usp=sharing)

## Import
`
from NitroFe import MovingAverageConvergenceDivergence
`

![MovingAverageConvergenceDivergence](https://media.giphy.com/media/YxGDsuUB8WkHnCnLXE/giphy.gif)

=
Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential
moving average (EMA) from the 12-period EMA.

The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line,"
is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals.

$$
Moving \ average \ convergence \ divergence \ (MACD) = Exponential \ moving \ average \ of \ (Absolute \ Price \ Oscillator)
$$

$$
Moving average \ convergence \ divergence \ histogram = (Absolute \ Price \ Oscillator)- (Exponential \ moving \ average \ of \ (Absolute \ Price))
$$


## Methods

::: NitroFE.time_based_features.indicator_features._MovingAverageConvergenceDivergence.MovingAverageConvergenceDivergence
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* investopedia, "Moving Average Convergence Divergence (MACD)",
    [https://www.investopedia.com/terms/m/macd.asp#:~:text=The%20MACD%20is%20calculated%20by,for%20buy%20and%20sell%20signals.](https://www.investopedia.com/terms/m/macd.asp#:~:text=The%20MACD%20is%20calculated%20by,for%20buy%20and%20sell%20signals.)