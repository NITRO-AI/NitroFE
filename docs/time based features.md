
# Time based Features

![Time based Features](https://media.giphy.com/media/xTk9Zx0YYJJqjZN4xa/giphy-downsized.gif)

Indicator / windows / moving averages features are dependent on past values for calculation, e.g. a rolling window of size 4 is dependent on past 4 values.

While creating such features during training is quite straighforward , taking it to production becomes challenging as it would requires one to externally save and implement logic. Creating indicators becomes even more complex as they are dependent on several other differently sized window components.

NitroFE internally handles saving past dependant values, and makes feature creation hassle free. Just use **fit_first=True** for your initial fit

NitroFE divides time based domain into 'Moving average features', 'Weighted window features' and 'indicator based features'

* [indicators features](indicators features.md)
* [moving average features](moving average features.md)
* [weighted window features](weighted window features.md)