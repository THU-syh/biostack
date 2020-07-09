__author__ = 'TramAnh'

from train import load_data

infile = 'Data/trainval.csv'
x_data, y_data = load_data(infile)

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

transformed = pca.fit_transform(x_data)