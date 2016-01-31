__author__ = 'zhoukai'


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

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


# def y(x):
#     return (12 + 3 * x) / 4.0
#
# plt.plot([0, -4, -5, 5], [3, 0, y(-5), y(5)])
# plt.axhline(0)
# plt.axvline(0)
# plt.ylim((-5, 5))
# plt.xlim((-5, 5))
#
# plt.savefig('3.png')



import struct
from array import array
from collections import defaultdict
import sys
from random import sample, randint
import numpy as np
import scipy.spatial


def load(path_img, path_lbl):
    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic))

        labels = array("B", file.read())

    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic))

        image_data = array("B", file.read())

    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    return images, labels


train_images, train_labels = load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
test_images, test_labels = load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")














