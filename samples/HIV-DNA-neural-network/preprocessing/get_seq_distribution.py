__author__ = 'TramAnh'

import csv
import pandas as pd
from collections import Counter

''' Examine a,t,c,g distribution in a sequence
    Purpose: to see which value to encode a,t,c,g,
    so that mean=0 and variance=1.
    This code doesn't show graph in pycharm,
    but works in iPython %matplotlib inline'''

infile = '../Data/train_cleaned.csv'

def show_distribution(arr):
    seq1, seq2 = arr
    seq1 = list(seq1)
    letter_counts = Counter(seq1)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df.plot(kind='bar')

if __name__ == '__main__':
    arr = []

    with open(infile, 'rb') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            arr.append([line[0], line[1]])

    input = 0
    while input != -1:
        show_distribution(arr[input])
