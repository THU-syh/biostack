"""
FILENAME:               processor03.py (/helical-project/structures/)
PROJECT:                HELiCAL
AUTHOR:                 Aakash "Kash" Sudhakar
DESCRIPTION:            Third algorithm structure for HELiCAL project data processing. 
                        Processor 03 enables reverse complement conversion of DNA strand.
DATE CREATED:           Thursday, March 21, 2019
DATE LAST MODIFIED:     Thursday, March 21, 2019
"""

class Processor03_ReverseComplement(object):
    """ Object structure containing logic for DNA reverse complement generation. """
    def __init__(self, data):
        self.dataset = data

    def reverse_complement_generator(self):
        """ Method to generate reversed complement strand (swapped base pair, reverse order) from DNA. """
        BASE_PAIRS = {
            "A": "T", 
            "T": "A", 
            "G": "C", 
            "C": "G"
        }
        return [str(value) for value in [BASE_PAIRS.get(key) for key in self.dataset.strip()]]

    def render_response(self, response):
        """ Method to render stylized response text to user. """
        return "\nOriginal genomic sequence: {}\n\nReverse complement sequence: {}".format(str(self.dataset), "".join(response)[::-1])