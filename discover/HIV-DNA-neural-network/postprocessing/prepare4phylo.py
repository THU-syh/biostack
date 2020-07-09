__author__ = 'TramAnh'

import csv

def write_to_phylo(outfile, data, type):
    '''
    Write to text file thats ready to be input in mafft phylo
    :param data: [sequence1, sequence2,...]
    :param type: either 'M' or 'W' for mutant and wildtype respectively
    :return: nothing
    '''
    with open(outfile, 'wb') as f:
        for i, each in enumerate(data):
            f.write('>%d%s\n'%(i+1, type))
            f.write(each+'\n')


if __name__ == '__main__':
    filepath1 = 'NN_prediction_summary.csv'
    filepath2 = 'set1_converted_summary.csv'
    testfile = '../Data/alignment/test_aligned.csv'

    # DNA level
    wt = []
    mt = []
    with open(testfile, 'rb') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            wt.append(line[0])
            mt.append(line[1])

    # AA level
    wt_aa = []
    mt_aa = []
    with open(filepath1, 'rb') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            if 'Wildtype' in line[0]:
                continue
            wt_aa.append(line[5])
            mt_aa.append(line[6])

    # write to file
    write_to_phylo(outfile='phylo_wt_dna.txt', data=wt, type='W')
    write_to_phylo(outfile='phylo_mt_dna.txt', data=mt, type='M')

    write_to_phylo(outfile='phylo_wt_aa.txt', data=wt_aa, type='W')
    write_to_phylo(outfile='phylo_mt_aa.txt', data=mt_aa, type='M')