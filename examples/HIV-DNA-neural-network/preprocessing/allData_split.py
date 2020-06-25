__author__ = 'TramAnh'

''' Split to train and test '''

from filter_input_data import filter, write_csv

dir = '/wobble_data'

filename = 'Data'+dir+'/trainvaltest.csv'

unseen_ratio = 0.15
test_ratio = 0.15

' Filter out noise sequence in trainvaltest.csv'
trainvaltest_filtered = filter(filename)        # trainvaltest_filtered = [[inp, out], [inp,out],...]


' Split data '
num_sample = len(trainvaltest_filtered)
traintest_mark = int((1-test_ratio-unseen_ratio)*num_sample)
testunseen_mark = int((1-unseen_ratio)*num_sample)

train = trainvaltest_filtered[:traintest_mark]
test = trainvaltest_filtered[traintest_mark:testunseen_mark]
unseen = trainvaltest_filtered[testunseen_mark:]

'Write to csv'
write_csv('Data'+dir+'/train_cleaned.csv', train)
write_csv('Data'+dir+'/test_cleaned.csv', test)
write_csv('Data'+dir+'/unseen_cleaned.csv', unseen)


