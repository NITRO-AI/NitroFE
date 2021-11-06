

# Class AbsolutePriceOscillator

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1NQ8j5kXuuONVHsjTs2MNLmfEfQR6id0S?usp=sharing)

## Import
`
from NitroFE import AbsolutePriceOscillator
`

![AbsolutePriceOscillator](https://media.giphy.com/media/CUFxB3z16aTJtXrsSW/giphy.gif)


The Absolute Price Oscillator displays the difference between two exponential moving averages

$$
Fast \ exponential \ moving \ feature \ (FEMF) = Exponential \ moving \ feature \ of \ 'fast\_period' \ span
$$

$$
Slow \ exponential \ moving \ feature \ (SEMF) = Exponential \ moving \ feature \ of \ 'slow\_period' \ span
$$

$$
Absolute \ Price \ Oscillator =  SEMF - FEMF
$$

## Methods

::: NitroFE.time_based_features.indicator_features._AbsolutePriceOscillator.AbsolutePriceOscillator
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* fidelity, "apo",
    [https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo)