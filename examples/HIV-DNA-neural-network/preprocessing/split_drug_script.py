__author__ = 'TramAnh'

import csv
import pandas as pd

infile = 'Data/alldata.csv'

# Read csv
alldata_df = pd.read_csv(infile, sep="\t")

# get unique drug name
unique_drug = alldata_df.Drug_D.unique()

# dictionary of drug as key and DF as value
DF_dict = { elem: pd.DataFrame for elem in unique_drug}

for key in DF_dict.keys():
    DF_dict[key] = alldata_df[:][alldata_df.Drug_D==key]
