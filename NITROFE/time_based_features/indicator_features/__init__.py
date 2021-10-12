
from NitroFE.time_based_features.indicator_features.indicator import absolute_price_oscillator,\
    moving_average_convergence_divergence,Average_true_range,average_directional_movement_index
    
#####################################################################

# ob = average_directional_movement_index()
# df = pd.DataFrame({"a": np.arange(25), "b": np.arange(25) + 2})
# df = pd.DataFrame(
#     {
#         "a": 10 * np.random.random(25),
#         "b": np.arange(0, 100, step=2)[:25]   + np.random.choice(np.arange(-20,20), size=25),
#     }
# )
# wz = 4
# mp = 2
# flp = 15
# print("df", df)
# res_all = ob.fit(
#                 df,
#                 first_fit=True,
#             directional_movement_lookback_period = wz,
#             directional_movement_min_periods = 1,

#             directional_movement_smoothing_period= 14,
#             directional_movement_smoothing_min_periods= 1,

#             average_directional_movement_smoothing_period= 14,
#             average_directional_movement_min_periods= 1,
#             )

# res_comb = pd.concat(
#     [
#         ob.fit(
#             df.iloc[:flp],
#             first_fit=True,
#             directional_movement_lookback_period = wz,
#             directional_movement_min_periods = 1,

#             directional_movement_smoothing_period= 14,
#             directional_movement_smoothing_min_periods= 1,

#             average_directional_movement_smoothing_period= 14,
#             average_directional_movement_min_periods= 1,
#         ),
#         ob.fit(df.iloc[flp:], first_fit=False),
#     ]
# )


# print(pd.concat([res_all, res_comb], axis=1))
# print((res_all.fillna(0)!=res_comb.fillna(0)).sum())

#####################################################################