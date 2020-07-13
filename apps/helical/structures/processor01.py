"""
FILENAME:               processor01.py (/helical-project/structures/)
PROJECT:                HELiCAL
AUTHOR:                 Aakash "Kash" Sudhakar
DESCRIPTION:            First algorithm structure for HELiCAL project data processing. 
                        Processor 01 enables frequency counting of nucleotides in DNA.
DATE CREATED:           Wednesday, March 20, 2019
DATE LAST MODIFIED:     Wednesday, March 20, 2019
"""

from collections import OrderedDict

class Processor01_NucleotideCounter(object):
    """ Object structure containing logic for DNA Nucleotide Counts processing algorithm. """
    def __init__(self, data):
        self.dataset = data

    def nucleotide_counter(self):
        """ Method to calculate frequency distribution of base occurrences in input data sequence. """
        dictogram, PERMITTED_NUCLEOTIDES = dict(), ["A", "C", "G", "T"]
        # Builds dictionary-structured histogram of nucleotide frequencies while checking for appropriate permitted nucleotides
        for nucleotide in self.dataset:
            if nucleotide in PERMITTED_NUCLEOTIDES:
                if nucleotide not in dictogram:
                    dictogram[nucleotide] = 1
                else:
                    dictogram[nucleotide] += 1
            continue
        # Creates ordered dictionary by key alphabetization and returns values in-line
        return OrderedDict(sorted(dictogram.items(), key=lambda X: X[0]))

    def render_response(self, response):
        """ Method to render stylized response text to user. """
        return " ".join(["{}".format(value) for key, value in response.items()])