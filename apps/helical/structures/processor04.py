"""
FILENAME:               processor04.py (/helical-project/structures/)
PROJECT:                HELiCAL
AUTHOR:                 Aakash "Kash" Sudhakar
DESCRIPTION:            Fourth algorithm structure for HELiCAL project data processing. 
                        Processor 04 enables determination of DNA GC-content.
DATE CREATED:           Thursday, March 21, 2019
DATE LAST MODIFIED:     Thursday, March 21, 2019
"""

class Processor04_GCContent(object):
    """ Object structure containing logic for strand-specific GC-content determination. """
    def __init__(self, data):
        self.dataset = data

    def gc_content_calculator(self):
        """ Calculates ratio of GC occurrences across entire DNA strand. """
        gc_content, DNA_LENGTH = int(), len(self.dataset)
        for base in self.dataset:            
                if base == "G" or base == "C":
                    gc_content += 1
        return round(float(100 * (gc_content / DNA_LENGTH)), 2)

    def render_response(self, response):
        """ Method to render stylized response text to user. """
        return "Original genomic sequence: {}\n\nRelative GC content: {}%".format(self.dataset, str(response))