import numpy as np
import matplotlib.pyplot as plt

n_data = 4
gess = {1.0, 3.8, 10.8, 4.3}
cwss = {1.1, 1.2, 0.8, 0.9}
mh = {1.8, 2, 1.9, 2.3}
hmc = {0.0, 2.0, 2.4, 8.2}

fig, ax = plt.subplots()
index = np.arange(n_data)
bar_width = 0.15
opacity = 0.8

rects1 = plt.bar(index, gess, bar_width,
alpha=opacity,
color='k',
label='GESS')

rects2 = plt.bar(index + bar_width, cwss, bar_width,
alpha=opacity,
color='b',
label='CWSS')

rects3 = plt.bar(index + bar_width*2, mh, bar_width,
alpha=opacity,
color='g',
label='MH')

rects4 = plt.bar(index + bar_width*3, hmc, bar_width,
alpha=opacity,
color='r',
label='HMC')

plt.xlabel('Datasets')
#plt.ylabel('Effective samples per second')
plt.title('Effective samples per second')
plt.xticks(index + bar_width, ('Funnel\n x8.5e+01', 'Mixture\n x1.9e+03', 'Breast Cancer\n x4.5e+01', 'German Credit\n x3.0e+02'))
plt.legend()

plt.tight_layout()
plt.show()
