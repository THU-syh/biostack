#!/bin/bash

# This will put the workflow of preprocessing and Neural network into a script
# Ultimate input is the drug name to split
# Output is acccuracy chart of different config of hiddennodes

DRUG="NFV"
# Get the specific drug
python ../preprocessing/filter_drug.py -d $DRUG -i ../Data/alldata_DrugSeq_pair.csv 

#Transform wobble characters
python ../wobble/transform_wobble.py -i $DRUG".csv" -o "traintest.csv"

# Prepare for alignment
python ../Alignment/convert_traintest_2fasta.py -i traintest.csv -o fasta4alignment.csv

# Pass to mafft
mafft fasta4alignment.csv > mafft_output.txt

# cut alignment
python ../Alignment/fasta_cut_alignment.py -i mafft_output.txt -o traintest_aligned.csv

# split train test
python ../preprocessing/trainval_split.py -i traintest_aligned.csv -r train.csv -e test.csv

# pass to NN
python ../ensemble/controller.py -r train.csv -e test.csv
