__author__ = 'zhoukai'

import struct
from array import array
import code
from collections import defaultdict
import math
import sys
from random import sample
import numpy as np


M = 1000
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


indexs = sample(range(len(train_images)), M)
train_images = np.array(train_images)[indexs]
train_labels = np.array(train_labels)[indexs]

# K-NN cluster (K = 10)
# cluster them according to label first

def center_cluster(l):
    len_list = len(l)
    len_pixels = len(l[0])
    center = [0 for i in range(len_pixels)]
    for node in l:
        for i in range(len_pixels):
            center[i] += node[i]
    center = [i * 1.0 / len_list for i in center]
    return center

def distance(o1, o2):
    dis = 0.0
    for i in range(len(o1)):
        dis += math.pow(o1[i] - o2[i], 2)
    return math.sqrt(dis)

def inList(node, list):
    for n in list:
        b = True
        for i in range(len(n)):
            if node[i] != n[i]:
                b = False
                break
        if b:
            return True
    return False

clusters = defaultdict(list)
for i in range(len(train_labels)):
    clusters[train_labels[i]].append(train_images[i])

for i in range(15):
    centers = [center_cluster(clusters[num]) for num in clusters]
    for num in clusters:
        clusters[num] = []
    for node in train_images:
        minv = sys.maxint
        num = -1
        for i in range(len(centers)):
            d = distance(centers[i], node)
            if d < minv:
                minv = d
                num = i
        clusters[num].append(node)
    print [(l, len(clusters[l])) for l in clusters]


trimmed_train_images = []
trimmed_train_labels = []
for i in range(len(train_images)):
    node = train_images[i]
    label = train_labels[i]
    if inList(node, clusters[label]):
        trimmed_train_images.append(node)
        trimmed_train_labels.append(label)

# test
def error_rate(train_images, train_labels, test_images, test_labels):
    error = 0
    for i in range(len(test_images)):
        print i
        test_node = test_images[i]
        test_label = test_labels[i]
        minv = sys.maxint
        index = -1
        for j in range(len(train_images)):
            train_node = train_images[j]
            d = distance(train_node, test_node)
            if d < minv:
                minv = d
                index = j
        if train_labels[index] != test_label:
            error += 1
    return error * 1.0 / len(test_images)

e1 =  error_rate(train_images, train_labels, test_images, test_labels)
# 0.16
e2 = error_rate(trimmed_train_images, trimmed_train_labels, test_images, test_labels)
# 0.17

print [(l, len(clusters[l])) for l in clusters]
print e1
print e2
print M