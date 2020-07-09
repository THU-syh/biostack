__author__ = 'TramAnh'

import numpy as np
import cPickle as pickle
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from utils import load_data
from sklearn.metrics import accuracy_score

def test_fn(testfile, hiddennodes, model_file):
    # load model
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

    def threshold(x):
        if x>0.5:
            print 'x>0.5'
        return 0 if x<0.5 else 1

    p_converted = []
    for each in p:
        converted = map(threshold, each)
        p_converted.append(converted)

    p_converted = np.array(p_converted)
    acc = accuracy_score(y_test, p_converted)
    print 'Accuracy score=%s' %acc



