# filter_sequences : function to filter the length of sequences
# INPUT:
#    1) sequences: list
#    2) length_minimum : minimum length acceptable 
#    3) length_maximum : maximum length acceptable 
def filter_sequences(sequences, length_minimum, length_maximum):
    # This function filters sequences based on their length. It takes a list of sequences and
    # two integers (length_minimum and length_maximum) as parameters.

    # Creates a list of indices for those sequences whose lengths are within the specified range.
    index = [i for i, seq in enumerate(sequences) if length_minimum <= len(seq) <= length_maximum]

    # Creates a list of sequences that are within the specified length range.
    sequences_valid = [seq for i, seq in enumerate(sequences) if length_minimum <= len(seq) <= length_maximum]

    # Returns two lists: one with the indices of the valid sequences and one with the valid sequences themselves.
    return index, sequences_valid


