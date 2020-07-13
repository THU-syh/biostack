"""
FILENAME:               app.py (/helical-project/)
PROJECT:                HELiCAL
AUTHOR:                 Aakash "Kash" Sudhakar
DESCRIPTION:            Main server setup file for running the HELiCAL project. 
DATE CREATED:           Wednesday, March 20, 2019
DATE LAST MODIFIED:     Wednesday, March 20, 2019
"""

# Imports Essential Assets from Flask
from flask import Flask, request, render_template

# Imports Custom Algorithm Structures for Data Processing
from .structures import processor01, processor02, processor03, processor04, processor05

# Initializes Flask App
app = Flask(__name__)

def _validate_data_general(dataset):
    """ Helper function to validate user input as genomic data. """
    PERMITTED_CHARS = ["A", "C", "G", "T"]
    for char in dataset:
        if char not in PERMITTED_CHARS:
            return -1

def _validate_data_codons(dataset):
    """ Helper function to validate user input as codon sequence. """
    if len(dataset) % 3 != 0:
        return -1

# Creates Home Page with User Data Input Option
@app.route("/")
def user_form():
    return render_template("input-data.html")

# Creates POST Route at Home Page with User Data Processing
@app.route("/", methods=["POST"])
def user_form_proc():
    # NOTE: PROCESSORS values reflect the _value_ tags in the HTML button elements in ./templates/input-data.html.
    PROCESSORS = {
        1: "Get Nucleotide Count",              # P01: DNA
        2: "Convert to RNA",                    # P02: RNA
        3: "Generate Reverse Complement",       # P03: REVC
        4: "Determine GC Content",              # P05: GC
        5: "Translate to Protein Chain"         # P08: PROT
    }
    if request.form["proc"] == PROCESSORS[1]:
        if request.form["proc01"]:
            text01 = request.form["proc01"].upper()
            if _validate_data_general(text01) != -1:
                proc01 = processor01.Processor01_NucleotideCounter(text01)
                return proc01.render_response(proc01.nucleotide_counter())
            else:
                return "ERROR: Inputted data is invalid. Please try again."
        else:
            return "ERROR: Nucleotide Counts Processor did not receive any data."
    elif request.form["proc"] == PROCESSORS[2]:
        if request.form["proc02"]:
            text02 = request.form["proc02"].upper()
            if _validate_data_general(text02) != -1:
                proc02 = processor02.Processor02_DNATransriber(text02)
                return proc02.render_response(proc02.transcriber())
            else:
                return "ERROR: Inputted data is invalid. Please try again."
        else:
            return "ERROR: RNA Transcriber Processor did not receive any data."
    elif request.form["proc"] == PROCESSORS[3]:
        if request.form["proc03"]:
            text03 = request.form["proc03"].upper()
            if _validate_data_general(text03) != -1:
                proc03 = processor03.Processor03_ReverseComplement(text03)
                return proc03.render_response(proc03.reverse_complement_generator())
            else:
                return "ERROR: Inputted data is invalid. Please try again."
        else:
            return "ERROR: Reverse Complement Processor did not receive any data."
    elif request.form["proc"] == PROCESSORS[4]:
        if request.form["proc04"]:
            text04 = request.form["proc04"].upper()
            if _validate_data_general(text04) != -1:
                proc04 = processor04.Processor04_GCContent(text04)
                return proc04.render_response(proc04.gc_content_calculator())
            else:
                return "ERROR: Inputted data is invalid. Please try again."
        else:
            return "ERROR: GC-Content Calculation Processor did not receive any data."
    elif request.form["proc"] == PROCESSORS[5]:
        if request.form["proc05"]:
            text05 = request.form["proc05"].upper()
            if _validate_data_general(text05) != -1 and _validate_data_codons(text05) != -1:
                preproc05 = processor02.Processor02_DNATransriber(text05)
                proc05 = processor05.Processor05_ProteinTranslator(preproc05.transcriber())
                return proc05.render_response(proc05.rna_to_protein_translator())
            else:
                return "ERROR: Inputted data is invalid or is not evenly divisible by three (3). Please try again."
        else:
            return "ERROR: Protein Chain Translator did not receive any data."
