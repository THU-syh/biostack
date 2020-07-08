__author__ = 'TramAnh'

from train import train_fn
from test import test_fn

if __name__=='__main__':
    hiddennodes = 6
    trainfile = '../Data/alignment/train_aligned.csv'
    testfile = '../Data/alignment/test_aligned.csv'

    model_file = 'Serialized/model_{0}_nodes.pkl'.format(str(hiddennodes))
    train_fn(trainfile,hiddennodes, model_file)
    print 'Done with training'
    test_fn(testfile, hiddennodes, model_file)
