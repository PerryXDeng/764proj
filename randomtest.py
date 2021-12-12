# # simple code to implement Runs
# # test of randomnes
# from matplotlib import pyplot as plt
# import random
# import math
# import statistics
#
# def runsTest(l, l_median):
#     runs, n1, n2 = 0, 0, 0
#
#     # Checking for start of new run
#     for i in range(len(l)):
#
#         # no. of runs
#         if (l[i] >= l_median and l[i - 1] < l_median) or \
#                 (l[i] < l_median and l[i - 1] >= l_median):
#             runs += 1
#
#             # no. of positive values
#         if (l[i]) >= l_median:
#             n1 += 1
#
#             # no. of negative values
#         else:
#             n2 += 1
#
#     runs_exp = ((2 * n1 * n2) / (n1 + n2)) + 1
#     stan_dev = math.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
#                          (((n1 + n2) ** 2) * (n1 + n2 - 1)))
#
#     z = (runs - runs_exp) / stan_dev
#
#     return z
#
#
# # Making a list of 100 random numbers
# zstat = []
# l = []
# for i in range(1000):
#     l.append(random.random())
#     if i > 3:
#         l_median = statistics.median(l)
#         Z = abs(runsTest(l, l_median))
#         zstat.append(Z)
#         print('Z-statistic= ', Z)
#
# plt.plot(zstat)
# plt.show()
