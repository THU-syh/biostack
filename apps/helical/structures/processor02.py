"""
FILENAME:               processor02.py (/helical-project/structures/)
PROJECT:                HELiCAL
AUTHOR:                 Aakash "Kash" Sudhakar
DESCRIPTION:            Second algorithm structure for HELiCAL project data processing. 
                        Processor 02 enables conversion (transcription) of DNA to RNA.
DATE CREATED:           Wednesday, March 20, 2019
DATE LAST MODIFIED:     Thursday, March 21, 2019
"""

class Processor02_DNATransriber(object):
    """ Object structure containing logic for DNA-to-RNA Transcription algorithm. """
    def __init__(self, data):
        self.dataset = data

    def transcriber(self):
        """ Method to transcribe relevant DNA base pairs (Thymine) into RNA base pairs (Uracil). """
        return "".join([[base, "U"][base == "T"] for base in self.dataset])

    def render_response(self, response):
        """ Method to render stylized response text to user. """
        return "\nOriginal genomic sequence: {}\n\nTranscribed genomic sequence: {}".format(str(self.dataset), response)