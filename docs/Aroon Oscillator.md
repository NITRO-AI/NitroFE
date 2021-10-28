
# Class AverageTrueRange

![AverageTrueRange](https://media.giphy.com/media/4ZsySSakheySRgatvI/giphy.gif)


The Aroon Oscillator is a trend-following indicator that uses aspects of the Aroon Indicator (Aroon Up and Aroon Down) to gauge the strength of a current trend and the likelihood that it will continue.

The Aroon Oscillator is calculated as 

$$
Aroon\_Up[t] = \frac{'lookback\_period'- Period \ since \ 'lookback\_period' \ High }{'lookback\_period'}
$$

$$
Aroon\_Down[t] = \frac{'lookback\_period'- Period \ since \ 'lookback\_period' \ Low }{'lookback\_period'}
$$

$$
Aroon\_Oscillator[t] = Aroon\_Up[t] - Aroon\_Down[t] 
$$

## Methods

::: NitroFE.time_based_features.indicator_features._AroonOscillator.AroonOscillator
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit

References
----------
* investopedia, "Aroon",
    [https://www.investopedia.com/terms/a/aroonoscillator.asp](https://www.investopedia.com/terms/a/aroonoscillator.asp)