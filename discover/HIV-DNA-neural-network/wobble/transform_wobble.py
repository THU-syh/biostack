__author__ = 'TramAnh'

import csv
import random
import sys, getopt

wobble_base = {}
wobble_base['b']= ('t') #('c', 'g', 't')       # t
wobble_base['d']= ('g') #('a', 'g', 't')       # g
wobble_base['h']= ('c') #('a', 'c', 't')       # c
wobble_base['k']= ('t') #('g', 't')            # t
wobble_base['m']= ('c') #('a', 'c')            # c
wobble_base['n']= ('t') #('a', 'c', 't', 'g')  # t
wobble_base['r']= ('a') #('a', 'g')            # a
wobble_base['s']= ('g') #('g', 'c')            # g
wobble_base['v']= ('a') #('a', 'c', 'g')       # a
wobble_base['w']= ('a') #('a', 't')            # a
wobble_base['y']= ('c') #('c', 't')            # c

valid = ('a', 'c', 't', 'g')

def convert(sequence):
    temp = ''
    for char in sequence:
        char = char.lower()
        if char in valid:
            temp += char
        else:
            new = random.choice(wobble_base[char])
            temp += new
            # print 'convert %s to %s' %(char, new)
    return temp

def main(argv):
    inputfile = ''  #'../Data/trainvaltest.csv'
    outputfile = '' #'../Data/wobble_data/trainvaltest.csv'
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
            inputfile = arg
            print 'inputfile={0}'.format(arg)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            print 'outputfile={0}'.format(arg)


    with open(inputfile,'rb') as f:
        arr = []
        csvreader = csv.reader(f)
        for line in csvreader:
            if 'seq_ND' in line:
                continue
            string1 = convert(line[0])
            string2 = convert(line[1])
            arr.append([string1, string2])

    # write to file
    with open(outputfile, 'wb') as writef:
        csvwriter = csv.writer(writef)
        csvwriter.writerows(arr)

if __name__=='__main__':
    main(sys.argv[1:])