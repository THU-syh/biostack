__author__ = 'TramAnh'

import csv
from filter_input_data import write_csv
import sys, getopt

def load_data(infile):
    arr = []
    with open(infile, 'rb') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            arr.append(line)

    return arr

def main(argv):
    infile = '' #'../Data/wobble_data/trainval_aligned.csv'
    trainfile = ''
    testfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:e:",["ifile=","trainfile=","testfile="])
    except getopt.GetoptError:
        print 'trainval_split.py -i <inputfile> -f1 <trainfile> -f2 <testfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'trainval_split.py -i <inputfile> -f1 <trainfile> -f2 <testfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            infile = arg
            print 'infile={0}'.format(arg)
        elif opt in ("-r", "--trainfile"):
            trainfile = arg
            print 'trainfile={0}'.format(arg)
        elif opt in ("-e", "--testfile"):
            testfile = arg
            print 'testfile={0}'.format(arg)

    trainval = load_data(infile)
    test_ratio = 0.15
    ' Split data '
    num_sample = len(trainval)
    trainval_mark = int((1-test_ratio)*num_sample)

    train = trainval[:trainval_mark]
    test = trainval[trainval_mark:]

    'Write to csv'
    write_csv(trainfile, train)
    write_csv(testfile, test)

if __name__=='__main__':
    main(sys.argv[1:])