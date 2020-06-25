__author__ = 'TramAnh'

import pandas as pd
import numpy as np
import csv
import cPickle as pickle

'''Process raw file from standforddb and take out sequence and drugs'''
# Read in PR.txt
f = open('../PR.txt', 'r')
flag = False
no_drug_lst = []
drug_lst = []
for line in f:
    if flag == False:
        flag = True
    else:
        tokens = line.split("\t")
        if tokens[-1].strip():
            ptid = int(tokens[1])
            drug = tokens[6]
            seq = tokens[-1].strip()
            if drug == 'None':
                no_drug_lst.append([ptid, drug, seq])
            else:
                drug_lst.append([ptid, drug, seq])

# Put into df to merge
no_drug_df = pd.DataFrame(no_drug_lst, columns=['ptid', 'Drug', 'seq'])
drug_df = pd.DataFrame(drug_lst, columns=['ptid', 'Drug', 'seq'])
merged = pd.merge(no_drug_df, drug_df, on='ptid', how='inner', suffixes=('_ND', '_D'))

merged.to_csv('Data/alldata.csv', sep="\t", index=False, quoting=csv.QUOTE_NONE)

trainvaltest = merged[['seq_ND','seq_D']]

trainvaltest.to_csv('Data/trainvaltest.csv', index=False, quoting=csv.QUOTE_NONE)



