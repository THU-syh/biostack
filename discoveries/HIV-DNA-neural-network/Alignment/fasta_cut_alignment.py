__author__ = 'TramAnh'

'''
Cut data according to alignment from mafft.
Input is data run from mafft (able to run in command line)
output is csv file of 2 columns, WildType (or ND) and Mutant (or Drug)
All data in output are of same length, and aligned
'''

import csv
from utils import put_seq_to_array
import sys, getopt

def get_sequence(infile, dict):
    with open(infile, 'rb') as f:
        value = ''
        key = ''
        for line in f:
            if '>' in line:     # Header '>..' line. Add (k,v) to dict and reset value and key
                if value and key:
                    dict[key] = value
                value = ''
                key = line.strip()[1:]
            else:
                value += line.strip()   # Concatenate value


def get_lcut_position(dict):
    half = len(dict[dict.keys()[0]])/2

    max_pos =0
    seq_name = ''
    for key in dict:
        half_seq = dict[key][:half]
        pos = half_seq.rfind('-')
        max_pos1 = max(pos, max_pos)
        seq_name = key if max_pos1 != max_pos else seq_name
        max_pos = max_pos1

    return seq_name, max_pos

def get_rcut_position(dict):
    half = len(dict[dict.keys()[0]])/2

    min_pos = half
    seq_name = ''
    for key in dict:
        half_seq = dict[key][half:]
        try:
            pos = half_seq.index('-')        # Return -1 if cant find
        except:
            pos = half
        min_pos1 = min(pos, min_pos)
        seq_name = key if min_pos1 != min_pos else seq_name
        min_pos = min_pos1

    rcut = half+ min_pos

    return seq_name, rcut

def write_arr_to_csv(arr, lcut, rcut, csv_file):
    with open(csv_file, 'wb') as f:
        csvwriter = csv.writer(f)
        for each in arr:
            wt = each[0][lcut+1:rcut]
            m = each[1][lcut+1:rcut]
            csvwriter.writerow([wt, m])

def main(argv):
    infile = '' #'../Data/wobble_data/fasta_output.txt'
    csv_file = ''   #'../Data/wobble_data/trainval_aligned.csv'

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
            infile = arg
            print 'infile={0}'.format(arg)
        elif opt in ("-o", "--ofile"):
            csv_file = arg
            print 'outfile={0}'.format(arg)

    dict= {}
    get_sequence(infile, dict)

    arr = put_seq_to_array(infile)

    print 'Cut left:'
    seq1, l_cut = get_lcut_position(dict)  # Return ['key', pos] => only consider the position
    print seq1, l_cut

    print '\nCut right:'
    seq2, r_cut = get_rcut_position(dict)
    print seq2, r_cut

    write_arr_to_csv(arr, l_cut, r_cut, csv_file)

if __name__=='__main__':
    main(sys.argv[1:])
