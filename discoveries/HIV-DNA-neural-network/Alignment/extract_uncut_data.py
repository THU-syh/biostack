__author__ = 'TramAnh'

from utils import put_seq_to_array, write_csv
import csv

infile = '../Data/alignment/fasta_output.txt'
trainvalfile = '../Data/alignment/trainval_uncut.txt'
trainfile = '../Data/alignment/train_uncut.txt'
testfile = '../Data/alignment/test_uncut.txt'

''' extract data straight from fasta output.
    Split data according to train:test ratio
    and write to file '''
if __name__=='__main__':
    trainval = put_seq_to_array(infile)

    with open(trainvalfile, 'wb') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(trainval)

    test_ratio = 0.15
    ' Split data '
    num_sample = len(trainval)
    trainval_mark = int((1-test_ratio)*num_sample)

    train = trainval[:trainval_mark]
    test = trainval[trainval_mark:]

    'Write to csv'
    write_csv(trainfile, train)
    write_csv(testfile, test)