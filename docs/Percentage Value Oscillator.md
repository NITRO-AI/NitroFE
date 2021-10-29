

# Class PercentageValueOscillator

## Import
`
from NitroFe import PercentageValueOscillator
`

![PercentageValueOscillator](https://media.giphy.com/media/pYs30P7XQcEZImCUIg/giphy.gif)

The percentage price oscillator (PPO) is a technical momentum indicator that shows the relationship between two moving averages in percentage terms. 

The percentage price oscillator (PPO) is calculated as 

$$
Fast \ exponential \ moving \ feature \ (FEMF) = Exponential \ moving \ feature \ of \ 'fast\_period' \ span
$$

$$
Slow \ exponential \ moving \ feature \ (SEMF) = Exponential \ moving \ feature \ of \ 'slow\_period' \ span
$$

$$
Raw \ Percentage \ Value \ Oscillator =  \frac{SEMF - FEMF}{FEMF}
$$

$$
Percentage \ Value \ Oscillator =  Exponential \ moving \ feature \ of \  'smoothing\_period' \ span \ over \ 'Raw \ Percentage \ Value \ Oscillator'
$$

## Methods

::: NitroFE.time_based_features.indicator_features._PercentageValueOscillator.PercentageValueOscillator
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* investopedia, "ppo",
    [https://www.investopedia.com/terms/p/ppo.asp](https://www.investopedia.com/terms/p/ppo.asp)