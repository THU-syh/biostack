__author__ = 'TramAnh'

import numpy as np
from sklearn.preprocessing import scale
from Levenshtein import distance
import pandas as pd
from Bio.Seq import Seq
from Bio.Alphabet import generic_rna

# encoder = {'c': -0.5, 't': -0.17, 'a':0.16, 'g':0.50}
# encoder = {'c': 1.0, 't': 2.0, 'a':3.0, 'g':4.0}
encoder = {'c': -1.0, 't': -0.5, 'a': 0.5, 'g': 1.0}
CODON_STR = 'Stop codon detected'

def __find_mutations_each(seq1, seq2):
    assert (len(seq1) == len(seq2))
    mut = {i for i in range(len(seq1)) if seq1[i] != seq2[i]}  # a set
    return mut


def find_mutation_pos(arr):
    '''~
    Param data: 2d list of sequences
    Take in data of 2 columns wildtype and mutant
    Return a set of all positions that ever have mutations occured
    '''
    a = set()

    x, y = arr
    for i in range(len(x)):
        a |= __find_mutations_each(x[i], y[i])  # |= is union operation
    return a


def load_data(arr):
    # [wt, mt] = arr      # wt and mt are list of sequence (in string)
    '''
    Convert array with actg to numbers
    '''

    def convert(wt):
        x_matrix = []
        for line in wt:
            temp = []
            for char in line:
                temp.append(encoder[char.lower()])
            x_matrix.append(temp)
        return x_matrix

    # Convert to numpy matrix
    x_matrix, y_matrix = [np.asarray(convert(x)) for x in arr]

    return x_matrix, y_matrix


def calculate_accuracy(arr1, arr2):
    '''arr1 and arr2 are mutant and predicted array of sequence. In string
        Return array of accuracy score'''

    accuracy_arr = map(accuracy, arr1, arr2)
    return accuracy_arr


def accuracy(str1, str2):
    ' Returns accuracy score from (0,1)'
    error = distance(str1, str2) / float(len(str1))
    return 1 - error

def dna_to_aa(sequence):
    ''' Convert DNA to Amino Acid sequence.
        Return converted_amino_acid ('Stop codon detected' if has stop codon)
    '''

    sequencelength = len(sequence)
    sample = Seq(sequence, generic_rna)
    amino_acid = sample.translate(to_stop=True).__str__()

    if len(amino_acid) < sequencelength/3:
        amino_acid = CODON_STR

    return amino_acid

def confusion_matrix(wt, mt, predicted):
    ' Calculate confusion matrix from list of strings wt, mt, predicted'

    data = [wt, mt, predicted]
    df = pd.DataFrame(data)
    df = df.transpose()

    cols = ['WT', 'MT', 'Predicted']
    df.columns = cols

    def row_op(x):
        wt = x['WT']
        mt = x['MT']
        predicted = x['Predicted']

        'Check if has stop codon'
        if predicted==CODON_STR or wt==CODON_STR or mt==CODON_STR:
            attr_list = [None]*7
            headers_list = ['Accuracy', '#Mutate Positions', 'TP', 'FN1', 'FN2', 'TN', 'FP']
            return pd.Series(attr_list, index=headers_list)

        score = accuracy(mt, predicted)

        tp, fp, fn1, fn2, tn = [0] * 5
        num_change = 0
        num_nochange = 0
        for i in range(len(wt)):
            w = wt[i].lower()
            m = mt[i].lower()
            p = predicted[i].lower()

            if m != w:
                if p == m:
                    tp += 1  # A -> T, predict A -> T
                elif p == w:
                    fn1 += 1  # A -> T, predict A -> A
                else:
                    fn2 += 1  # A -> T, predict A -> G
                num_change += 1
            else:  # case m == w
                if p == m:
                    tn += 1  # A -> A, predict A -> A
                elif p != m:
                    fp += 1  # A -> A, predict A -> T
                num_nochange += 1

        MAXLENGTH = len(wt)
        try:
            tp = tp / float(num_change)
            fn1 = fn1 / float(num_change)
            fn2 = fn2 / float(num_change)
        except ZeroDivisionError:
            tp, fn1, fn2 = [None] * 3

        try:
            tn = tn / float(num_nochange)
            fp = fp / float(num_nochange)
        except ZeroDivisionError:
            tn, fp = [None] * 2

        attr_list = [score, num_change, tp, fn1, fn2, tn, fp]
        headers_list = ['Accuracy', '#Mutate Positions', 'TP', 'FN1', 'FN2', 'TN', 'FP']

        return pd.Series(attr_list, index=headers_list)

    df1 = df.apply(row_op, axis=1).round(3)  # Round to 3dp
    df = df.merge(df1, left_index=True, right_index=True)

    return df
