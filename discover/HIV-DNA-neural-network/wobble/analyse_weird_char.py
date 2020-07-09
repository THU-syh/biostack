__author__ = 'TramAnh'

''' Analyse the position of the weird character. What it was before or what it mutated from'''

import csv

filename = 'Data/alldata.csv'

def filter(filename):
    output_anomaly = []
    with open(filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")     # delimiter is already ,
        for row in csvreader:
            if 'ptid' in row:
                continue
            invalid_analysis(row, output_anomaly)
                # write to file
    return output_anomaly

def invalid_analysis(row, output):
    valid = ('a', 'c', 't', 'g', '-')

    ptid = row[0]
    drug = row[3]
    seq_ND = row[2]
    seq_D = row[4]

    for i in range(max(len(seq_ND), len(seq_D))):
        try:
            char_ND = seq_ND[i].lower()
        except:
            char_ND = '-'

        try:
            char_D = seq_D[i].lower()
        except:
            char_D = '-'

        if char_ND not in valid or char_D not in valid:
            output.append([ptid, drug, (char_ND,i), (char_D, i)])

with open('Data/anomaly.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter="\t")
    csvwriter.writerow(['ptid', 'drug', 'seq_ND', 'seq_D'])
    output = filter(filename)
    csvwriter.writerows(output)