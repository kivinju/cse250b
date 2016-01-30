__author__ = 'zhoukai'

'''
this is an slightly different algorithm.
instead of remove the nodes which are not in the cluster, we only remain the nodes which are not in the cluster

'''

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
    return scipy.spatial.distance.euclidean(o1, o2)
    # dis = 0.0
    # for i in range(len(o1)):
    #     dis += math.pow(o1[i] - o2[i], 2)
    # return math.sqrt(dis)


def inList(node, list):
    for n in list:
        if np.array_equal(n, node):
            return True
    return False


clusters = defaultdict(list)
for i in range(len(train_labels)):
    clusters[train_labels[i]].append(train_images[i])

print [(l, len(clusters[l])) for l in clusters]

for i in range(20):
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

f = open('incluster_output.txt', 'a+')
f.write(", ".join([str(len(clusters[l])) for l in clusters]) + "\n")
f.flush()


def run_alg(M):
    indexs = sample(range(len(train_images)), M)
    random_train_images = np.array(train_images)[indexs]
    random_train_labels = np.array(train_labels)[indexs]

    count = 0
    indexs = []
    for i in range(len(train_images)):
        if count >= M:
            break
        node = train_images[i]
        label = train_labels[i]
        if inList(node, clusters[label]):
            print i, count
            count += 1
            indexs.append(i)

    while count < M:
        index = randint(0, len(train_images) - 1)
        if index not in indexs:
            print count
            count += 1
            indexs.append(index)

    trimmed_train_images = np.array(train_images)[indexs]
    trimmed_train_labels = np.array(train_labels)[indexs]

    e1 = error_rate(random_train_images, random_train_labels, test_images, test_labels)
    e2 = error_rate(trimmed_train_images, trimmed_train_labels, test_images, test_labels)

    print "random : " + e1
    print "trimmed : " + e2

    f.write("random : " + str(e1) + "\n")
    f.write("trimmed : " + str(e2) + "\n")
    f.write("num : " + M + "\n")
    f.flush()

run_alg(1000)
run_alg(5000)
run_alg(10000)


f.close()