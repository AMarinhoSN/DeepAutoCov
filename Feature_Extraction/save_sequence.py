# validate_sequences: function to find the correct sequences
def validate_sequences(sequences):
    valid_sequences = []
    invalid_sequences = []
    valid_indices = []
    invalid_indices = []
    for index, seq in enumerate(sequences):
        is_valid = True
        for amino_acid in seq:
            if amino_acid not in "ACDEFGHIKLMNPQRSTVWY":
                is_valid = False
                break
        if is_valid:
            valid_sequences.append(seq)
            valid_indices.append(index)
        else:
            invalid_sequences.append(seq)
            invalid_indices.append(index)
    return valid_sequences, invalid_sequences, valid_indices, invalid_indices


