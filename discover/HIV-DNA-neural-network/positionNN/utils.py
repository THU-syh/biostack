__author__ = 'TramAnh'

import csv
import numpy as np

# encoder = {'c': -1.0, 't': -0.5, 'a': 0.5, 'g': 1.0}
encoder = {'c': [1,0,0,0], 't':[0,1,0,0], 'a':[0,0,1,0], 'g':[0,0,0,1]}
def encode(char):
    return encoder[char.lower()]

def compare(a, b):
    return int(a!=b)

def load_data(filename):
    x_data = []
    y_data = []
    with open(filename, 'rb') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            wt = row[0]
            mt = row[1]

            wt_num = map(encode, wt)    # Convert to number
            wt_num = sum(wt_num, [])
            x_data.append(wt_num)

            y = map(compare, wt, mt)
            y_data.append(y)
    return np.array(x_data), np.array(y_data)

# if __name__=='__main__':
#     load_data('../Data/train_cleaned.csv')