def remove_asterisks(sequence):
    # This function is designed to remove asterisks (*) from the end of a given sequence.
    # Parameter:
    # sequence: A string representing the sequence from which asterisks should be removed.

    # The function uses Python's rstrip method on the sequence.
    # rstrip("*") removes all trailing asterisks (*) from the end of the sequence.
    # If there are no asterisks at the end, the sequence remains unchanged.
    return sequence.rstrip("*")


