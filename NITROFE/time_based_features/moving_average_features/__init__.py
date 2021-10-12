
from NitroFE.time_based_features.moving_average_features.moving_average_features import exponential_moving_feature,\
                                            hull_moving_feature,calculate_triple_exponential_moving_feature
                                            
#####################################################################

# ob = exponential_moving_feature()
# df = pd.DataFrame({"a": np.arange(14), "b": np.arange(14) + 2})
# df = pd.DataFrame(
#     {
#         "a": np.arange(25) + 10 * np.random.random(25),
#         "b": np.concatenate((np.arange(10),np.array([np.nan]),np.arange(11,25))),
#     }
# )
# wz = 6
# mp=2
# print(df)
# res_all = ob.fit(df, first_fit=True, span=wz,min_periods=mp,ignore_na=False)


# res_comb = pd.concat(
#     [
#         ob.fit(df.iloc[:12], first_fit=True, span=wz,min_periods=mp,ignore_na=False),
#         ob.fit(df.iloc[12:], first_fit=False),
#     ]
# )



# print(pd.concat([res_all, res_comb], axis=1))

#####################################################################