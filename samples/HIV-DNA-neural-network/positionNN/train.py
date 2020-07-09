__author__ = 'TramAnh'

import numpy as np
import csv
import cPickle as pickle
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, TanhLayer
from utils import load_data

def train_fn(trainfile, hiddennodes, output_model_file):

    hidden_size = hiddennodes

    print 'Loading data..'
    x_train, y_train = load_data(trainfile)

    input_size = x_train.shape[1]
    target_size = y_train.shape[1]

    # prepare dataset

    ds = SDS( input_size, target_size )
    ds.setField( 'input', x_train )
    ds.setField( 'target', y_train )

    # init and train

    net = buildNetwork(input_size, hidden_size, target_size, bias = True, hiddenclass=SigmoidLayer,
                       outclass=SigmoidLayer)
    trainer = BackpropTrainer(net,ds)

    print 'Training..'
    trainer.trainUntilConvergence(validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )

    print 'Finish training. Serializing model...'
    pickle.dump( net, open( output_model_file, 'wb' ))