from utils import dna_to_aa, confusion_matrix
import csv
from Levenshtein import distance
from os import path
import os


def alert(has_stop):
    if has_stop:
        print 'Stop codon detected'

def summary(predictfile, actualfile, outfile):
    predict_dna = []
    predict_aa = []
    actual_wt_dna = []
    actual_wt_aa = []
    actual_mt_dna = []
    actual_mt_aa = []

    tp_dna, fn1_dna, fn2_dna, tn_dna, fp_dna = [[] for _ in range(5)]
    tp_aa, fn1_aa, fn2_aa, tn_aa, fp_aa = [[] for _ in range(5)]

    # Distance
    dist_dna = []
    dist_aa = []

    CODON_MSG = 'Contains stop codon'

    # Analyse prediction
    with open(predictfile, 'rb') as f:
        for line in f:
            dna = line.strip()
            predict_dna.append(dna)
            has_stop, aa = dna_to_aa(dna)     # dna_to_aa returns [has_stop_codon, aa]
            alert(has_stop)
            if has_stop:
                aa = CODON_MSG
            predict_aa.append(aa)

    # Analyse clinical data
    with open(actualfile, 'rb') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            wt = line[0]
            mt = line[1]
            actual_wt_dna.append(wt)
            actual_mt_dna.append(mt)

            # Convert to Amino Acid
            has_stop, wt_aa = dna_to_aa(wt)
            alert(has_stop)
            if has_stop:
                wt_aa = CODON_MSG
            actual_wt_aa.append(wt_aa)

            has_stop, mt_aa = dna_to_aa(mt)
            alert(has_stop)
            if has_stop:
                mt_aa = CODON_MSG
            actual_mt_aa.append(mt_aa)

    # Calculate confusion matrix
    for i in range(len(actual_wt_dna)):
        a, b, c, d, e = confusion_matrix(wt=actual_wt_dna[i],
                                          mt=actual_mt_dna[i],
                                          predicted=predict_dna[i])
        tp_dna.append(a)
        fn1_dna.append(b)
        fn2_dna.append(c)
        tn_dna.append(d)
        fp_dna.append(e)

        try:
            a,b,c,d,e = confusion_matrix(wt=actual_wt_aa[i], mt=actual_mt_aa[i], predicted=predict_aa[i])
        except TypeError:
            print 'debug'
        tp_aa.append(a)
        fn1_aa.append(b)
        fn2_aa.append(c)
        tn_aa.append(d)
        fp_aa.append(e)

        MAXLENGTH = len(actual_mt_dna[i])
        dist_dna.append(distance(actual_mt_dna[i].lower(), predict_dna[i].lower())/float(MAXLENGTH))
        dist_aa.append(distance(actual_mt_aa[i].lower(), predict_aa[i].lower())/float(MAXLENGTH))

    # Calculate Levenstein distance


    with open(outfile, 'wb') as f:
        csvwriter = csv.writer(f)
        headers = ['Wildtype (DNA)', 'Mutant (DNA)', 'Predicted (DNA)',
                   'TP (DNA)', 'FN1 (DNA)', 'FN2 (DNA)', 'TN (DNA)', 'FP (DNA)', 'ErrorRate (DNA)',

                   'Wildtype (AA)', 'Mutant (AA)', 'Predicted (AA)',
                   'TP (AA)', 'FN1 (AA)', 'FN2 (AA)', 'TN (AA)', 'FP (AA)', 'ErrorRate (AA)']
        csvwriter.writerow(headers)

        num_seq = len(predict_dna)
        for i in range(num_seq):
            row = [actual_wt_dna[i], actual_mt_dna[i], predict_dna[i],
                   tp_dna[i], fn1_dna[i], fn2_dna[i], tn_dna[i], fp_dna[i], dist_dna[i],

                   actual_wt_aa[i], actual_mt_aa[i], predict_aa[i],
                   tp_aa[i], fn1_aa[i], fn2_aa[i], tn_aa[i], fp_aa[i], dist_aa[i]]
            csvwriter.writerow(row)

if __name__== '__main__':
    path = '/Users/TramAnh/Dropbox/NTU/BII_Parttime_work/HIV/Summary/'
    actualfile = path+'test_aligned.csv'    # WT, MT in DNA
    predictpath = path+'Output_data/'       # in DNA
    predictfiles = [predictpath+f for f in os.listdir(predictpath)]
    for predictfile in predictfiles:
        if 'DS' in predictfile:
            continue
        outfile = predictfile[:-4]+'_summary.csv'
        print predictfile
        print actualfile
        print outfile
        summary(predictfile, actualfile, outfile)



