#import argparse
import os
import csv
# import external libraries
import pandas as pd
import statistics as st

# --- I/O functions ---

def read_fasta(file):
    sequences = []
    with open(file, 'r') as f:
        current_sequence = ''
        started = False
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if started:
                    sequences.append(current_sequence)
                started = True
                current_sequence = ''
            else:
                current_sequence += line
        if current_sequence:
            sequences.append(current_sequence)
    return sequences



class feature_extractor:
    """
    """
    def __init__(self, fasta_file, metadata_csv) -> None:
        # read fasta
        self.sequences = read_fasta(fasta_file)
        # read csv
        self.metadata_arr = pd.read_csv(metadata_csv).values
    
    # -- handle sequence methods
    def remove_chars_seq(self, char="*"):
        self.sequences = [s.rstrip(char) for s in self.sequences]
    
    def validate_sequences(self,sequences):
        """
        validate_sequences: function to find the correct sequences
        """

        VALID_AA = "ACDEFGHIKLMNPQRSTVWY"

        valid_sequences = []
        invalid_sequences = []
        valid_indices = []
        invalid_indices = []

        for index, seq in enumerate(sequences):
            is_valid = True
            for amino_acid in seq:
                if amino_acid not in VALID_AA:
                    is_valid = False
                    break
            if is_valid:
                valid_sequences.append(seq)
                valid_indices.append(index)
            else:
                invalid_sequences.append(seq)
                invalid_indices.append(index)
        
        # --- update attributes ---
        self.valid_sequences = valid_sequences
        self.invalid_sequences = invalid_sequences
        self.valid_indices = valid_indices
        self.invalid_indices = invalid_indices
    