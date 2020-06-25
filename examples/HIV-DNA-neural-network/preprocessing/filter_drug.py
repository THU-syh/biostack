__author__ = 'TramAnh'

import sys, getopt
import csv
# print 'Hello world from command line! {0}'.format(sys.argv)
import pandas as pd

def main(argv):
    drug = ''
    datapath = '' #'Data/alldata_DrugSeq_pair.csv'

    try:
        opts, args = getopt.getopt(argv,"hi:d:",["ifile=", "drug="])
    except getopt.GetoptError:
        print 'file.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'file.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            datapath = arg
            print 'infile={0}'.format(arg)
        elif opt in ("-d", "--drug"):
            drug = arg
            print 'drug={0}'.format(arg)

    df = pd.read_csv(datapath, sep='\t')
    drugdf = df[df['Drug_D']==drug][['seq_ND','seq_D']]
    drugdf.to_csv(drug+'.csv',index=False, quoting=csv.QUOTE_NONE)

if __name__=='__main__':
    main(sys.argv[1:])