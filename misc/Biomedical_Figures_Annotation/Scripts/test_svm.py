#!/usr/bin/python3

import os
import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import argparse
import matplotlib.pyplot as plt
#import pandas as pd
import pickle
from train_svm import load_image, loadModel
import cv2 as cv
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input


PATH       = '/home/dimitris/GitProjects/Biomedical_Figure_Annotator/model.pkl'
#TEST_PATH = '/home/dimitris/GitProjects/Biomedical_Figure_Annotator/test_data.csv'
TEST_PATH  = '/media/dimitris/TOSHIBA EXT/Image_Document_Classification/PMC-Dataset/' 
TRAIN_PATH = '/home/dimitris/GitProjects/Biomedical_Figure_Annotator/train_data.csv'

def save_predictions():
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Path arguments')
    parser.add_argument('-mp', '--model_path', nargs=1)
    parser.add_argument('-tp', '--test_path', nargs=1)
    parser.add_argument('-l', '--layer', nargs='+')

    return parser.parse_args()

def load_svm(path):
    svm = joblib.load(path)
    return svm

def check_args(args):
    if not os.path.exists(args.model_path[0]) and os.path.exists(args.testPath[0]):
        raise Exception('Missing Paths: \n === \n model Path: {} \n test Path {}')
        sys.exit(-1)

def load_data(path):
    data_frame = pd.read_csv(path, header=None)
    return data_frame[0].tolist(), np.array(data_frame[1].values)

def load_train_data(path):
    data_frame = pd.read_csv(path, header=None)
    return data_frame[0].tolist()

def remove_train(trainList, test):
    for figure in trainList:
        if figure in test:
            test.remove(figure)
    return test

def save_to_csv(paths, predictions):
    # manual initialized with the condition of whether the figures 
    # have been corrected
    manual  = np.zeros(len(paths), dtype=int)    
    cid     = pd.DataFrame(manual)
    figures = pd.DataFrame(paths)
    pred    = pd.DataFrame(predictions)

    frames  = [figures, pred, cid]
    result  = pd.concat(frames, axis=1)
    result.to_csv('predictions.csv', header=False, index=False)

def image_retrieval(path):
    imageList = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            select = os.path.join(subdir, file)
            #print(select)
            if '.jpg' in select:
                imageList.append(select)
    return imageList


def main(args):
    feats = np.array([])
    #print(args)
    #print(args.model_path)
    #print(type(args.model_path))
    check_args(args)

    svm_model = load_svm(args.model_path[0])   
    # Evaluate
    #figures, labels = load_data(args.test_path[0])
    
    trainList = load_train_data(TRAIN_PATH)
    testList  = image_retrieval(TEST_PATH)
    print(len(testList))
    figures   = remove_train(trainList, testList)

    print(len(figures)) 
    print(len(testList))
    #sys.exit(1)
    model = loadModel('vgg')
    
    if args.layer:
        layer = args.layer[0]
        intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer).output)            
        intermediate_layer_model.summary()

    for image in figures:
        figure = load_image(image)
        
        if args.layer:
            y_pred = intermediate_layer_model.predict(figure)
        else:    
            y_pred = model.predict(figure, verbose=1)
        
        if feats.size == 0:
            feats = y_pred
        else:
            feats = np.concatenate((feats, y_pred), axis=0)
    
    #feats = np.max(feats, axis=-1)
    feats = np.reshape(feats, (feats.shape[0],-1))
    predictions = svm_model.predict(feats)
    
    """
    score = accuracy_score(labels, predictions)
    print('Score {}'.format(score))
    print('Predictions {}'.format(predictions))
    print('labels {}'.format(labels))
    """
    save_to_csv(figures, predictions)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    sys.exit(1)
