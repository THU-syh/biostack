__author__ = 'TramAnh'

from train import load_data
import numpy as np
import cPickle as pickle
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from scipy.stats.stats import pearsonr
from Levenshtein import distance
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
import matplotlib.pyplot as plt
import csv

output_predictions_file = 'predictions.txt'

def test_fn(testfile, hiddennodes):
    # load model
    model_file = '../Serialized/model_{0}_nodes.pkl'.format(str(hiddennodes))
    net = pickle.load( open( model_file, 'rb' ))
    print 'Finish loading model'

    # Load test data
    x_test, y_test = load_data(testfile)
    y_test_dummy = np.zeros( y_test.shape )
    input_size = x_test.shape[1]
    target_size = y_test.shape[1]

    assert( net.indim == input_size )
    assert( net.outdim == target_size )

    # prepare dataset
    ds = SDS( input_size, target_size )
    ds.setField( 'input', x_test )
    ds.setField( 'target', y_test_dummy )

    # predict
    print 'Activating ds'
    p = net.activateOnDataset( ds )

    dna_converted = decode_whole(p)
    actual_dna = decode_whole(y_test)
    r_score = calculate_errorrate(actual_dna, dna_converted)

    # Write converted predicted sequence to file
    with open('../results_log/prediction.csv', 'wb') as f:
        for item in dna_converted:
            print>>f, item

    return r_score


def calculate_errorrate(y_test, p):
    r = []
    MAXLENGTH = len(y_test[0])
    for i in range(len(p)):
        # r_score = pearsonr(y_test[i], p[i])
        distance_score = 1.0 - distance(y_test[i], p[i])/float(MAXLENGTH)
        r.append(distance_score)        # pearsonr returns correlation and 2-tailed p-value. Only need correlation
    return r

def decode_whole(p):
    '''
    Decode from real sequence to threshold sequence
    aka convert from number to DNA
    :param p: predicted sequence (in list of sequence)
    :return: list of threshold values
    '''
    texts = []
    for seq in p:
        decoded_intext = decode_each(seq)
        texts.append(decoded_intext)

    return texts

def decode_each(x):
    'param x is a list of numbers'
    # assert (type(x) is float)
    temp = ''

    # Convert to characters
    length = len(x)
    i = 0
    while (i < length):
        seq = x[i:i+4].tolist()
        max_value = max(seq)
        max_index = seq.index(max_value)
        encode_rule = ['a', 't', 'c', 'g']
        temp += encode_rule[max_index]
        i += 4

    return temp

if __name__=='__main__':
    test_fn()