"""
Analyze the results obtained from the experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join

DATA_DIR = 'data'

DATA_FILE = join(DATA_DIR, 'passenger_statistics.csv')

passengers = pd.read_csv(DATA_FILE)

passengers['system_time'] = passengers['waiting_time'] + passengers['boarding_time']
passengers['threshold'] = passengers['waiting_time'] > 60
print(passengers.head())
print(passengers.describe())

grouped = passengers.groupby('episode').filter(lambda x: len(x) > 50).groupby('episode')
# print(grouped)
# print(grouped.groups)
stats = grouped.aggregate(np.mean)
plt.plot(stats['waiting_time'])
plt.show()
# plt.figure()
# sns.distplot(passengers['waiting_time'], kde=False)
# plt.figure()
# sns.distplot(passengers['boarding_time'], kde=False)
# plt.show()