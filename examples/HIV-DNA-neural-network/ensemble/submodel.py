__author__ = 'TramAnh'

import cPickle as pickle
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, SigmoidLayer
from pybrain.structure.modules   import SoftmaxLayer
import numpy as np
from utils import load_data, encoder
from sklearn import preprocessing

class SubModel:
    '''
    Class for Mutation and NoMutation model training
    '''
    def __init__(self, hiddennodes, type):
        self.hiddennodes = hiddennodes
        self.type = type
        self.model_file = 'Serialized/bundle_{0}_nodes_{1}.pkl'.format(str(self.hiddennodes),self.type)
            # bundle because it consists of [net, std_scale]


    def train(self, arr):
        '''
        Train NN for given data
        :param arr: [wt_arr, mt_arr], in ATCG or atcg
        :return: void, but serialize model to file
        '''

        x_train, y_train = load_data(arr)
        std_scale = preprocessing.StandardScaler().fit(x_train)
        x_train_scaled = std_scale.transform(x_train)     # Normalize to standard normal
        # y_train_scaled = std_scale.transform(y_train)     # Try not scaling y

        input_size = x_train_scaled.shape[1]
        target_size = y_train.shape[1]

        # prepare dataset

        ds = SDS( input_size, target_size )
        ds.setField( 'input', x_train_scaled )
        ds.setField( 'target', y_train )

        # init and train

        net = buildNetwork(input_size, self.hiddennodes, target_size, bias = True, hiddenclass=TanhLayer,
                           outclass=TanhLayer)
        trainer = BackpropTrainer(net,ds)

        print 'Training..'
        trainer.trainUntilConvergence(validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )

        print 'Finish training. Serializing bundle...'
        bundle = [net, std_scale]
        pickle.dump( bundle, open( self.model_file, 'wb' ))

    def test(self, arr):
        # load model
        net, std_scale = pickle.load( open( self.model_file, 'rb' ))
        print 'Finish loading model'

        # Load test data
        x_test, y_test = load_data(arr)
        x_test_scaled = std_scale.transform(x_test)       # Normalize to standard normal

        y_test_dummy = np.zeros( y_test.shape )
        input_size = x_test_scaled.shape[1]
        target_size = y_test.shape[1]

        assert( net.indim == input_size )
        assert( net.outdim == target_size )

        # prepare dataset
        ds = SDS( input_size, target_size )
        ds.setField( 'input', x_test_scaled )
        ds.setField( 'target', y_test_dummy )

        # predict
        print 'Activating ds'
        p = net.activateOnDataset( ds )
        print 'debug'
        # ptest = preprocessing.StandardScaler().fit_transform(p)
        # p_scaled = std_scale.inverse_transform(ptest)  # Convert back to original scale

        dna = self.convert_to_dna(p)

        return dna

    def convert_to_dna(self, p):
        '''Convert numerical results to nearest dna threshold'''
        arr = []
        for seq in p:
            temp = ''
            for num in seq:
                dist = [abs(num-encoder[x]) for x in encoder.keys()]
                value = encoder.keys()[dist.index(min(dist))]
                temp += value
            arr.append(temp)
        return arr