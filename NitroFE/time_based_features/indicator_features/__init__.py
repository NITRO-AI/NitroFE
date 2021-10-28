
import numpy as np
import pandas as pd
from NitroFE.time_based_features.indicator_features._AbsolutePriceOscillator import AbsolutePriceOscillator
from NitroFE.time_based_features.indicator_features._PercentageValueOscillator import PercentageValueOscillator
from NitroFE.time_based_features.indicator_features._MovingAverageConvergenceDivergence import MovingAverageConvergenceDivergence
from NitroFE.time_based_features.indicator_features._AverageTrueRange import AverageTrueRange
from NitroFE.time_based_features.indicator_features._AverageDirectionalMovementIndex import AverageDirectionalMovementIndex
from NitroFE.time_based_features.indicator_features._AroonOscillator import AroonOscillator
from NitroFE.time_based_features.indicator_features._TypicalValue import TypicalValue
from NitroFE.time_based_features.indicator_features._BollingerBands import BollingerBands
from NitroFE.time_based_features.indicator_features._KaufmanEfficiency import KaufmanEfficiency
from NitroFE.time_based_features.indicator_features._TripleExponentialMovingAverageOscillator import TripleExponentialMovingAverageOscillator
from NitroFE.time_based_features.indicator_features._ZeroLagExponentialMovingFeature import ZeroLagExponentialMovingFeature
from NitroFE.time_based_features.indicator_features._RelativeStrengthIndex import RelativeStrengthIndex
from NitroFE.time_based_features.indicator_features._SeriesWeightedAverage import SeriesWeightedAverage
from NitroFE.time_based_features.indicator_features._SeriesWeightedMovingFeature import SeriesWeightedMovingFeature
from NitroFE.time_based_features.indicator_features._InverseFisherRelativeStrengthIndex import InverseFisherRelativeStrengthIndex
from NitroFE.time_based_features.indicator_features._KeltnerChannel import KeltnerChannel




# print("#####################################################################")

# ob = KeltnerChannel()
# df = pd.DataFrame(
#     {
#         "a": 10 * np.random.random(20),
#         "c": np.arange(0, 100, step=2)[:20]
#         + np.random.choice(np.arange(0, 20), size=20),
#     }
# )
# df2 = pd.DataFrame(
#     {
#         "a": 10 * np.random.random(20),
#         "b": np.arange(0, 100, step=2)[:20]
#         + np.random.choice(np.arange(0, 20), size=20),
#     }
# )

# wz = 4
# mp = 1
# flp = 8
# flp2 = 15

# res_all = ob.fit(df[:],first_fit=True).reset_index(drop=True)

# print("#####################################################################")
# res_comb = pd.concat(
#     [
#         ob.fit(df.iloc[:flp], first_fit=True),
#         ob.fit(df.iloc[flp:flp2],  first_fit=False),
#         ob.fit(df.iloc[flp2:],  first_fit=False),
#     ]
# ).reset_index(drop=True)

# print(pd.concat([res_all, res_comb], axis=1))
# print((res_all.fillna(0) != res_comb.fillna(0)).sum())
# print("#####################################################################")



#####################################################################