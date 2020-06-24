#!/usr/bin/python

from __future__ import print_function
import keras
import numpy as np
import pandas as pd
import sys
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg19 import VGG19
from keras import backend as K
import read_activations 
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.svm import SVC
from sklearn.externals import joblib

PATH_1 = '/media/dimitris/TOSHIBA EXT/Image_Document_Classification/PMC-Dataset/'
PATH_2 = '/media/dimitris/TOSHIBA EXT/Image_Document_Classification/PMC-Dataset/PMC5777379/ijn-13-439Fig2.jpg'
PATH   = '/home/dimitris/GitProjects/Biomedical_Figure_Annotator/train_data.csv'
img_rows = 600
img_cols = 800
channels = 3

def plot_classified_images(data):
    plt.scatter(data, '*')
    plt.show()

def cluster_features(feats):
    feats = np.reshape(feats, (feats.shape[0], -1))
    assert feats.ndim == 2, print(feats.shape)
    #Kmeans      = KMeans(n_clusters=7, random_state=0).fit(feats)
    #KMeans(n_clusters=7, random_state=0).fit(feats)
    #predictions = Kmeans.transform(feats)
    print(feats.shape)
    embed_feats = TSNE(n_components=4, random_state=0).fit_transform(feats)
    return np.amax(embed_feats, axis=-1)

def load_image(imagePath, show=False, scale=True, sampleSize=(800,600)):
    img = image.load_img(imagePath)
    img = image.img_to_array(img)
    #
    if scale:
        img = cv.resize(img, dsize=sampleSize, interpolation=cv.INTER_CUBIC)
        img = np.clip(img, 0, 255)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    if show:
        plt.imshow(img[0]/255.)                           
        #plt.axis('off')
        plt.show()
    return img


def upsampling(image):
    '''
    Takes as input the image and returns the upsampled image
    '''
    pass

def image_retrieval(path):
    imageList = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            select = os.path.join(subdir, file)
            #print(select)
            if '.jpg' in select:
                imageList.append(select)
    print(len(imageList))
    #print(imageList)
    return imageList

def loadModel(parameter):
    if parameter.lower() == 'vgg':
        model = VGG19(include_top=False, input_shape=(img_rows, img_cols, channels), pooling=None, weights='imagenet')
    elif parameter.lower() == 'resnet':
        model = ResNet50(include_top=False, input_shape=(img_rows, img_cols, channels), pooling=None, weights='imagenet')
    else:
        print('Unknown Argument {}'.format(parameter))
        sys.exit(-1)
    return model

def layerNames(model):
    return [layers.name for layers in model.layers]

def parseArguments():
    # Define parser
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-n', '--network', nargs=1)
    parser.add_argument('-l', '--layer', nargs='+')
    return parser.parse_args()

def train_svm(feats, labels, C_coef=12.):
    print(feats.shape)
    svm = SVC(C=C_coef).fit(feats, labels)
    return svm

def save_model(model):
    joblib.dump(model, 'model.pkl')

def load_data(path):
    data_frame = pd.read_csv(path, header=None)#, index_col=0)
    return data_frame[0].tolist(), np.array(data_frame[1].values)

def main(args):
    figures = np.array([])
    
    paths, labels = load_data(PATH)
    model = loadModel(args.network[0])
    model.summary()

    if args.layer:
        layer = args.layer[0]
        intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer).output)            
        intermediate_layer_model.summary()

    for file in paths:
        print(file)
        figure = load_image(file, show=False)
        
        # select the proper model object
        if args.layer:
            y_pred = intermediate_layer_model.predict(figure)
        else:    
            y_pred = model.predict(figure, verbose=1)

        # check dimensions
        if figures.size == 0:
            figures = np.expand_dims(y_pred, axis=0)
        else:
            figures = np.concatenate((figures, np.expand_dims(y_pred, axis=0)), axis=0)
    
    # Fix this
    print(figures.shape)
    figures = np.reshape(figures, (figures.shape[0], -1))
    svm_plane = train_svm(figures, np.array(labels))   
    save_model(svm_plane)

if __name__ == '__main__':
    args = parseArguments()
    main(args)