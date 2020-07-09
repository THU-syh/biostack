__author__ = 'TramAnh'

from mainmodel import MainModel
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys, getopt

def plot_histogram(r, hiddennodes):
    plt.subplot(140+hiddennodes-2)
    plt.hist(r, bins=20)
    plt.title("hiddennodes=%s" %hiddennodes)
    # plt.ylim(0,10)

def mapper(hiddennodes):
    print 'Running hiddennodes %s' %(hiddennodes)
    model = MainModel(hiddennodes)
    print 'Training..'
    model.train(trainfile=trainfile)
    print 'Testing..'
    score = model.test(testfile=testfile)
    print score
    return hiddennodes, score

def main():
    # Parallel run
    # plt.figure(1)
    # pool = Pool(4)
    # out= pool.map(mapper, range(3,7))
    #
    # for i in out:
    #     nodes = i[0]
    #     plot_histogram(i[1], nodes)
    #
    # plt.show()

    # Single run
    mapper(6)

if __name__=='__main__':
    trainfile='../Data/wobble_data/train_aligned.csv'
    testfile='../Data/wobble_data/test_aligned.csv'

    'For arguments in shell script'
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hr:e:",["trainfile=","testfile="])
    except getopt.GetoptError:
        print 'trainval_split.py -i <inputfile> -r <trainfile> -e <testfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'trainval_split.py -i <inputfile> -r <trainfile> -e <testfile>'
            sys.exit()
        elif opt in ("-r", "--trainfile"):
            trainfile = arg
            print 'trainfile={0}'.format(arg)
        elif opt in ("-e", "--testfile"):
            testfile = arg
            print 'testfile={0}'.format(arg)

    main()