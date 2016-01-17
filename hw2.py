#!/usr/bin/env python

from collections import defaultdict
import numpy as np
import math
import sys


def read_data_file(filename):
    data_file = open(filename)
    dict = {}
    for line in data_file:
        l = line.split()
        docIdx = eval(l[0])
        wordIdx = eval(l[1])
        count = eval(l[2])
        if docIdx not in dict:
            dict[docIdx] = defaultdict(int)
        dict[docIdx][wordIdx] += count
    data_file.close()
    return dict


def read_data_label(filename):
    data_file = open(filename)
    dict = defaultdict(int)
    map = defaultdict(int)

    count = 0
    for line in data_file:
        groupId = eval(line)
        dict[groupId] += 1
        count += 1
        map[count] = groupId
    data_file.close()
    return dict, map

voc_file = open("data/vocabulary.txt")
voc_dict = defaultdict(int)
count = 0
for line in voc_file:
    count += 1
    voc_dict[line] = count
voc_file.close()

train_data = read_data_file("data/train.data")
train_label, train_map = read_data_label("data/train.label")

group_num = len(train_label)
doc_num = len(train_map)
voc_num = len(voc_dict)

pi = [train_label[groupId] * 1.0 / doc_num for groupId in range(1, 21)]
# smoothing
p = np.ones((group_num, voc_num))

for docId in train_data:
    for vId in train_data[docId]:
        p[train_map[docId] - 1][vId - 1] += train_data[docId][vId]

for groupId in range(len(p)):
    group_sum = sum(p[groupId])
    for vId in range(len(p[groupId])):
        p[groupId][vId] = p[groupId][vId] / group_sum



# routine groupId: 1 - 20
def helper(data, groupId):
    result = 0.0
    result += math.log(pi[groupId - 1])
    for wordId in data:
        result += data[wordId] * math.log(p[groupId - 1][wordId - 1])
    return result

def choose(data):
    m = -sys.maxint - 1
    result = 0
    for groupId in range(1, 21):
        temp = helper(data, groupId)
        if temp > m:
            m = temp
            result = groupId
    return result

test_data = read_data_file("data/test.data")
test_label, test_map = read_data_label("data/test.label")

test_num = len(test_map)
error_num = 0
for docId in test_map:
    if choose(test_data[docId]) != test_map[docId]:
        error_num += 1

error_rate = error_num * 1.0 / test_num
print "error_rate" + str(error_rate)

