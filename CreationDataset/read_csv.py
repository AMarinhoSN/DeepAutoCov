import pandas as pd

def read_csv(file):
    return pd.read_csv(file).values

def read_tsv(file):
    return pd.read_csv(file,sep='\t').values