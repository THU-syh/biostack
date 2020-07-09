__author__ = 'TramAnh'

from Bio.Seq import Seq
from Bio.Alphabet import generic_rna
import numpy as np

def dna_to_aa(sequence):
    ''' Convert DNA to Amino Acid sequence.
        Return [has_stop_codon, converted_amino_acid]
    '''
    if len(sequence) != 225:
        print len(sequence)

    sample = Seq(sequence, generic_rna)
    amino_acid = sample.translate(to_stop=True).__str__()

    codon = False
    if len(amino_acid) < 225/3:
        'Stop codon detected'
        codon = True

    return codon, amino_acid

def confusion_matrix(wt, mt, predicted):
    '''
    Take in a wildtype, mutant and predicted sequence.
    String type.
    Compute a confusion matrix of
                predicted
                NC          C
    actual  NC
            C
    '''

    if 'stop codon' in predicted:
        return [None] * 5

    # Make sure same length
    assert(len(wt) == len(mt))
    assert(len(mt) == len(predicted))

    tp, fp, fn1, fn2, tn= [0] * 5
    num_change = 0
    num_nochange = 0
    for i in range(len(wt)):
        w = wt[i].lower()
        m = mt[i].lower()
        p = predicted[i].lower()

        if m != w:
            if p == m:
                tp += 1     # A -> T, predict A -> T
            elif p == w:
                fn1 += 1    # A -> T, predict A -> A
            else:
                fn2 += 1    # A -> T, predict A -> G
            num_change += 1
        else:   # case m == w
            if p == m:
                tn += 1     # A -> A, predict A -> A
            elif p != m:
                fp += 1     # A -> A, predict A -> T
            num_nochange += 1

    MAXLENGTH = len(wt)
    try:
        tp = tp/float(num_change)
        fn1 = fn1/float(num_change)
        fn2 = fn2/float(num_change)
    except ZeroDivisionError:
        tp, fn1, fn2 = [0] * 3

    try:
        tn = tn/float(num_nochange)
        fp = fp/float(num_nochange)
    except ZeroDivisionError:
        tn, fp = [0] * 2

    return tp, fn1, fn2, tn, fp

