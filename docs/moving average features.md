
# Moving average features

![exponential_moving_feature](https://media.giphy.com/media/iYTXFJofI7I987H92k/giphy-downsized-large.gif)

In statistics, a moving average (rolling average or running average) is a calculation to analyze data points by creating a series of averages of different subsets of the full data set.

## Available moving average features

Moving averages utilize past values for calculation of the current values, hence while it might be easy to create such features during training in a single go, productionalizing or implementation for test become a challenge.

NitroFE features overcome this challenge by internally saving the past dependant values (the number of to be saved depend upon the logic of the feature ).
For your training/initial fit phase use fit_first=True, and for any production/test implementation pass fit_first=False

* [exponential moving average](exponential moving average.md)
* [hull moving average](hull moving average.md)
* [kaufman adaptive moving average](kaufman adaptive moving average.md)
* [fractal adaptive moving average](fractal adaptive moving average.md)
* [triple exponential moving average](triple exponential moving average.md)
* [smoothed moving average](smoothed moving average.md)