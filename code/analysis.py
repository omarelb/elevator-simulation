"""
Analyze the results obtained from the experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import BSpline, make_interp_spline

from os.path import join

DATA_DIR = 'data'

DATA_FILE = join(DATA_DIR, 'stats.csv')
dtypes = {'episode': int, 'waiting_time': np.float32, 'boarding_time': np.float32}
# passengers = pd.read_csv(DATA_FILE, dtype=dtypes, sep=',', low_memory=True)
passengers = pd.read_csv(DATA_FILE, dtype=dtypes)
x = np.arange(len(passengers)) * 2
spl = make_interp_spline(x, passengers['waiting_time'], k=3)
plt.plot(x, spl(x))
plt.show()
quit()
# x = passengers.info(memory_usage='deep')
# print(x)
# passengers['system_time'] = passengers['waiting_time'] + passengers['boarding_time']
# passengers['threshold'] = passengers['waiting_time'] > 60
# print(passengers.head())
# print(passengers.describe())

# grouped = passengers.groupby('episode')#.filter(lambda x: len(x) > 50).groupby('episode')
# print(grouped
# print(grouped.groups)
# stats = grouped.aggregate(np.mean)
# print(stats)
# stats.to_csv('stats.csv')
plt.plot(passengers['waiting_time'])
plt.show()
plt.figure()
sns.distplot(passengers['waiting_time'], kde=False)
plt.figure()
sns.distplot(passengers['boarding_time'], kde=False)
plt.show()