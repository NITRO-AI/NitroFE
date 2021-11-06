

# Class PercentageValueOscillator

* Jump right in for a hands-on [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1LWdYv9ITjf1nUPCGgCB_JoOT0bTv6U5C?usp=sharing)

## Import
`
from NitroFE import PercentageValueOscillator
`

![PercentageValueOscillator](https://media.giphy.com/media/OyycyPJomAv6YVXCtV/giphy.gif)

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