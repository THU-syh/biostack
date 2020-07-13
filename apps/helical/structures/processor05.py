"""
FILENAME:               processor05.py (/helical-project/structures/)
PROJECT:                HELiCAL
AUTHOR:                 Aakash "Kash" Sudhakar
DESCRIPTION:            Fourth algorithm structure for HELiCAL project data processing. 
                        Processor 05 enables translation of RNA to protein sequence.
DATE CREATED:           Thursday, March 21, 2019
DATE LAST MODIFIED:     Thursday, March 21, 2019
"""

class Processor05_ProteinTranslator(object):
    """ Object structure containing logic for DNA-to-RNA-to-protein genomic translation. """
    def __init__(self, data):
        self.dataset = data
        self.TRANSLATION_TABLE = {
            "UUU": "F",      "CUU": "L",      "AUU": "I",      "GUU": "V",
            "UUC": "F",      "CUC": "L",      "AUC": "I",      "GUC": "V",
            "UUA": "L",      "CUA": "L",      "AUA": "I",      "GUA": "V",
            "UUG": "L",      "CUG": "L",      "AUG": "M",      "GUG": "V",
            "UCU": "S",      "CCU": "P",      "ACU": "T",      "GCU": "A",
            "UCC": "S",      "CCC": "P",      "ACC": "T",      "GCC": "A",
            "UCA": "S",      "CCA": "P",      "ACA": "T",      "GCA": "A",
            "UCG": "S",      "CCG": "P",      "ACG": "T",      "GCG": "A",
            "UAU": "Y",      "CAU": "H",      "AAU": "N",      "GAU": "D",
            "UAC": "Y",      "CAC": "H",      "AAC": "N",      "GAC": "D",
            "UAA": "Stop",   "CAA": "Q",      "AAA": "K",      "GAA": "E",
            "UAG": "Stop",   "CAG": "Q",      "AAG": "K",      "GAG": "E",
            "UGU": "C",      "CGU": "R",      "AGU": "S",      "GGU": "G",
            "UGC": "C",      "CGC": "R",      "AGC": "S",      "GGC": "G",
            "UGA": "Stop",   "CGA": "R",      "AGA": "R",      "GGA": "G",
            "UGG": "W",      "CGG": "R",      "AGG": "R",      "GGG": "G"
        }

    def rna_to_protein_translator(self):
        """ Method to convert transcribed DNA into corresponding protein sequence. """
        protein_seq = list()
        for iterator in range(0, len(self.dataset), 3):
            codon = self.dataset[iterator:iterator+3]
            if self.TRANSLATION_TABLE[codon] == "Stop":
                return protein_seq
            protein_seq.append(self.TRANSLATION_TABLE[codon])
        return protein_seq
    
    def render_response(self, response):
        """ Method to render stylized response text to user. """
        return "Transcribed RNA sequence: {}\n\nTranslated protein chain: {}".format(self.dataset, "-".join(response))