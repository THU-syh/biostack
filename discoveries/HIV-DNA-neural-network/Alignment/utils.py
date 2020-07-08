__author__ = 'TramAnh'

import csv

def put_seq_to_array(infile):
    arr = []
    with open(infile, 'rb') as f:
        id=''
        wt_seq = ''
        m_seq = ''
        temp = ''
        for line in f:
            line = line.strip()
            if '>' in line:
                if 'W' in line:
                    if id:
                        m_seq = temp
                        arr.append([wt_seq, m_seq])
                        temp = ''
                if 'M' in line:
                    wt_seq = temp
                    id = line[1:-1]     # line is in form '>10M'
                    temp = ''
            else:
                temp += line.strip()
    return arr

def write_csv(filename, output):
    with open(filename, 'wb') as csvwrite:
        csvwriter = csv.writer(csvwrite)
        csvwriter.writerows(output)