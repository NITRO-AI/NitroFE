

# Class AbsolutePriceOscillator

## Import
`
from NitroFe import AbsolutePriceOscillator
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