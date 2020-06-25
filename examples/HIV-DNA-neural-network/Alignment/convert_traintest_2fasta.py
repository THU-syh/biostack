__author__ = 'TramAnh'

import csv
import sys, getopt

def get_sequences(filename, arr):
    with open(filename, 'rb') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            arr.append(line)

def write_fasta(outfile, arr):
    with open(outfile, 'wb') as f:
        for i, line in enumerate(arr):
            f.write('>{0}W\n'.format(i))
            f.write(line[0].strip()+'\n')
            f.write('>{0}M\n'.format(i))
            f.write(line[1].strip()+'\n')

def main(argv):
    file = '../Data/wobble_data/trainvaltest.csv'
    outfile = '../Data/wobble_data/fasta_4alignment.txt'

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'transform_wobble.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'transform_wobble.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file = arg
            print 'inputfile={0}'.format(arg)
        elif opt in ("-o", "--ofile"):
            outfile = arg
            print 'outputfile={0}'.format(arg)

    arr=[]
    get_sequences(file, arr)
    write_fasta(outfile, arr)

if __name__=='__main__':
    main(sys.argv[1:])

