![zoofs Logo Header](asserts/rescaled_logo.jpeg)

# NitroFE ( Nitro Feature Engineering )

[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=jaswinder9051998_zoofs&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=NITRO-AI_NitroFE)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=jaswinder9051998_zoofs&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=NITRO-AI_NitroFE)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=jaswinder9051998_zoofs&metric=security_rating)](https://sonarcloud.io/dashboard?id=NITRO-AI_NitroFE)

``NitroFE`` is a Python feature engineering engine which provides a variety of feature engineering modules designed to handle continuous calculation.

## Documentation
https://nitro-ai.github.io/NitroFE/
 
## Installation
[![PyPi version](https://badgen.net/pypi/v/NitroFE/)](https://pypi.com/project/NitroFE)

### Using pip

Use the package manager to install NitroFE.

```bash
pip install NitroFE
```

# Available feature domains

* Jump right in with an introduction [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1LDM9er9x7NJogRlHkRcLB4bhU1mK0l-9?usp=sharing)

## [Time based Features](https://nitro-ai.github.io/NitroFE/Time%20based%20features/)

![Time based Features](https://media.giphy.com/media/xTk9Zx0YYJJqjZN4xa/giphy-downsized.gif)

Indicator / windows / moving averages features are dependent on past values for calculation, e.g. a rolling window of size 4 is dependent on past 4 values.

While creating such features during training is quite straighforward , taking it to production becomes challenging as it would requires one to externally save and implement logic. Creating indicators becomes even more complex as they are dependent on several other differently sized window components.

NitroFE internally handles saving past dependant values, and makes feature creation hassle free. Just use **first_fit=True** for your initial fit

The Time based domain is divided into 'Moving average features', 'Weighted window features' and 'indicator based features'

## [Indicators based Features](https://nitro-ai.github.io/NitroFE/indicators%20features/)

![Time based Features](https://media.giphy.com/media/XfmGjCqHUVSrTrfxIU/giphy.gif)

NitroFe provides a rich variety of features which are inspired and translated from market indicators.
* [Absolute Price Oscillator](https://nitro-ai.github.io/NitroFE/Absolute%20Price%20Oscillator/)
* [Percentage price oscillator](https://nitro-ai.github.io/NitroFE/Percentage%20Value%20Oscillator/)
* [Moving average convergence divergence](https://nitro-ai.github.io/NitroFE/Moving%20Average%20Convergence%20Divergence/)
* [Average true range](https://nitro-ai.github.io/NitroFE/Average%20True%20Range/)
* [Average directional index](https://nitro-ai.github.io/NitroFE/Average%20Directional%20Movement%20Index/)
* [Aroon Oscillator](https://nitro-ai.github.io/NitroFE/Aroon%20Oscillator/)
* [Bollinger Bands](https://nitro-ai.github.io/NitroFE/Bollinger%20Bands/)
* [Kaufman Efficiency](https://nitro-ai.github.io/NitroFE/Kaufman%20Efficiency/)
* [Triple Exponential Moving Average Oscillator](https://nitro-ai.github.io/NitroFE/Kaufman%20Efficiency/)
* [Zero lag exponential moving average](https://nitro-ai.github.io/NitroFE/Zero%20Lag%20Exponential%20Moving%20Feature/)
* [Relative Strength Index](https://nitro-ai.github.io/NitroFE/Relative%20Strength%20Index/)
* [Inverse Fisher Relative Strength Index](https://nitro-ai.github.io/NitroFE/Inverse%20Fisher%20Relative%20Strength%20Index/)
* [Series Weighted Average](https://nitro-ai.github.io/NitroFE/Series%20Weighted%20Average/)
* [Series Weighted Moving Feature](https://nitro-ai.github.io/NitroFE/Series%20Weighted%20Moving%20Feature/)
* [Keltner channel](https://nitro-ai.github.io/NitroFE/Keltner%20Channel/)

## [Moving average features](https://nitro-ai.github.io/NitroFE/moving%20average%20features/)

![exponential_moving_feature](https://media.giphy.com/media/t7sEnf5w7wJ1CEPyy7/giphy.gif)

In statistics, a moving average (rolling average or running average) is a calculation to analyze data points by creating a series of averages of different subsets of the full data set. NitroFE provides an array of variety of moving averages type for you to utilize.

* [Exponential moving average](https://nitro-ai.github.io/NitroFE/exponential%20moving%20average/)
* [Hull Moving Average](https://nitro-ai.github.io/NitroFE/hull%20moving%20average/)
* [Kaufman's Adaptive Moving Average](https://nitro-ai.github.io/NitroFE/kaufman%20adaptive%20moving%20average/)
* [Fractal Adaptive Moving Average](https://nitro-ai.github.io/NitroFE/fractal%20adaptive%20moving%20average/)
* [Triple exponential moving average](https://nitro-ai.github.io/NitroFE/triple%20exponential%20moving%20average/)
* [Smoothed Moving Average](https://nitro-ai.github.io/NitroFE/smoothed%20moving%20average/)

## [Weighted window Features](https://nitro-ai.github.io/NitroFE/weighted%20window%20features/)

NitroFe provides easy to use functions to creatind weighted window features 
