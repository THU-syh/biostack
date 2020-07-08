__author__ = 'TramAnh'

import train
import test
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
score=[]
testfile = '../Data/alignment/test_aligned.csv'
trainfile = '../Data/alignment/train_aligned.csv'

def plot_histogram(r, hiddennodes):
    plt.subplot(140+hiddennodes-2)
    plt.hist(r, bins=20)
    plt.title("hiddennodes=%s" %hiddennodes)
    # plt.ylim(0,10)

def mapper(hiddennodes):
    print 'Running hiddennode=%d' %(hiddennodes)
    # train.train_fn(trainfile, hiddennodes)
    r_score = test.test_fn(testfile, hiddennodes)
    # score.append([hiddennodes, score_test])R
    print 'Length of samples= %s'%len(r_score)
    print 'Hidden nodes= {0}    Score_test= {1}'.format(hiddennodes, r_score)
    return hiddennodes, r_score

# Parallel run
plt.figure(1)
pool = Pool(4)
out= pool.map(mapper, range(3,7))

for i in out:
    nodes = i[0]
    plot_histogram(i[1], nodes)

plt.show()

# Single run
# mapper(5)
