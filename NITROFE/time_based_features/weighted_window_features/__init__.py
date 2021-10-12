
import numpy as np
import pandas as pd
from NitroFE.time_based_features.weighted_window_features.weighted_window_features import weighted_window_features

##################################################################################

# df = pd.DataFrame({"a": np.random.random(14), "b": np.random.random(14) * 10})
# print('df',df)
# from scipy import signal
# ss=False
# ww=4*2
# fl=2
# mp=2
# def perc1(data, pp):
#     return np.percentile(data, pp)

# ob = weighted_window_features()
# res_all = ob.caluclate_equal_feature(
#     dataframe=df,first_fit= True,window=ww, min_periods=mp, operation=np.mean )

# res_comb = pd.concat(
#     [
#         ob.caluclate_equal_feature(
#             dataframe=df.iloc[:fl],first_fit=True ,window=ww,min_periods= mp, operation=np.mean),
#         ob.caluclate_equal_feature(dataframe=df.iloc[fl:], first_fit=False),
#     ]
# )

# print(pd.concat([res_all, res_comb], axis=1))

##################################################################################