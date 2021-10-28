

# Class TripleExponentialMovingAverageOscillator

## Import
`
from NitroFe import TripleExponentialMovingAverageOscillator
`

![TripleExponentialMovingAverageOscillator](https://media.giphy.com/media/wOC7i7lyqRwZkhnawu/giphy.gif)


The Triple Exponential Moving Average Oscillator calculates the rate of change of Triple Exponential Moving Average

$$
Triple\_Exponential\_Moving\_Average\_Oscillator[t] = \frac{Triple\_Exponential\_Moving\_Average[t]-Triple\_Exponential\_Moving\_Average[t-1]}{Triple\_Exponential\_Moving\_Average[t-1]}
$$


## Methods

::: NitroFE.time_based_features.indicator_features._TripleExponentialMovingAverageOscillator.TripleExponentialMovingAverageOscillator
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* investopedia, "triple-exponential-moving-average",
    [https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp](https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp)