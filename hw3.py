__author__ = 'zhoukai'


import numpy as np
import matplotlib.pyplot as plt

# mean = [0, 0]
# cov = [[9, 0], [0, 1]]
#
# x, y = np.random.multivariate_normal(mean, cov, 100).T
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.savefig('2a.png')

mean = [0, 0]
cov = [[1, -0.75], [-0.75, 1]]

x, y = np.random.multivariate_normal(mean, cov, 100).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.savefig('2b.png')