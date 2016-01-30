__author__ = 'zhoukai'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# mean = [0, 0]
# cov = [[9, 0], [0, 1]]
#
# x, y = np.random.multivariate_normal(mean, cov, 100).T
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.savefig('2a.png')


# mean = [0, 0]
# cov = [[1, -0.75], [-0.75, 1]]
#
# x, y = np.random.multivariate_normal(mean, cov, 100).T
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.savefig('2b.png')


def y(x):
    return (12 + 3 * x) / 4.0

plt.plot([0, -4, -5, 5], [3, 0, y(-5), y(5)])
plt.axhline(0)
plt.axvline(0)
plt.ylim((-5, 5))
plt.xlim((-5, 5))

plt.savefig('3.png')

