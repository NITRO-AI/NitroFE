

# Class TripleExponentialMovingAverageOscillator

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1bG9qIgToV5QvNJMOzi001Snm62TdFXyj?usp=sharing)

## Import
`
from NitroFE import TripleExponentialMovingAverageOscillator
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