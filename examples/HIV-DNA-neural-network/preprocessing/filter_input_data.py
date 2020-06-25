__author__ = 'TramAnh'

'''Filter those noise data in trainval and test.csv'''

import csv

def filter(filename):
    output = []
    with open(filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)     # delimiter is already ,
        for row in csvreader:
            if not check_for_valid(row):
                continue
            output.append(row)
    return output

def check_for_valid(row):
    valid = ('a', 'c', 't', 'g')
    for each in row:
        for char in each:
            if char.lower() not in valid:
                return False
    return True

def write_csv(filename, output):
    with open(filename, 'wb') as csvwrite:
        csvwriter = csv.writer(csvwrite)
        csvwriter.writerows(output)

if __name__=='__main__':
    trainval = 'Data/trainval.csv'
    test = 'Data/test.csv'

    # output filename
    out_trainval_file = 'Data/trainval_cleaned.csv'
    out_test_file = 'Data/test_cleaned.csv'

    # array of the cleaned data
    output_trainval = filter(trainval)
    output_test = filter(test)

    write_csv(out_trainval_file, output_trainval)
    write_csv(out_test_file, output_test)




