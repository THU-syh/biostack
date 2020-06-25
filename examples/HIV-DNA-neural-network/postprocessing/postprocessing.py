__author__ = 'TramAnh'

import csv

def postproc_fn():
    f = open('predictions.txt', 'rb')
    p_sequences = []
    for line in f:
        nums = line.split()
        temp = ''
        for num in nums:
            temp = temp + decode(float(num))
        p_sequences.append(temp)

    # Compare with test.csv
    # TODO: fix index out of bound error
    # elaborate: predictions.txt only includes 20 sequences
    # while test.csv has a lot more than 20 sequences
    # Reason: test.csv can have a lot of data, but only 20 are valid (that only contains ATCG)
    similar = []
    with open('test.csv', 'rb') as testfile:
        reader = csv.reader(testfile,)
        for i, row in enumerate(reader):
            d = row[1]
            o = p_sequences[i]
            similar.append(similarity(d, o))

    mean = sum(similar)/len(similar)
    print 'Mean similarity= %f' %(mean)

if '__name__'=='__main__':
    postproc_fn()
