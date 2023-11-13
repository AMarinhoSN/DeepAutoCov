def validate_sequences(sequences):
    # Initializing lists to store valid and invalid sequences, and their respective indices.
    valid_sequences = []
    invalid_sequences = []
    valid_indices = []
    invalid_indices = []

    # Iterating over each sequence and its index in the provided list of sequences.
    for index, seq in enumerate(sequences):
        # Initially assuming the sequence is valid.
        is_valid = True

        # Checking each amino acid in the sequence.
        for amino_acid in seq:
            # If the amino acid is not one of the standard amino acids, mark the sequence as invalid.
            if amino_acid not in "ACDEFGHIKLMNPQRSTVWY":
                is_valid = False
                break  # Exiting the loop as soon as an invalid amino acid is found.

        # If the sequence is valid, add it and its index to the valid lists.
        if is_valid:
            valid_sequences.append(seq)
            valid_indices.append(index)
        else:
            # If the sequence is invalid, add it and its index to the invalid lists.
            invalid_sequences.append(seq)
            invalid_indices.append(index)

    # Returning the lists of valid and invalid sequences, along with their indices.
    return valid_sequences, invalid_sequences, valid_indices, invalid_indices