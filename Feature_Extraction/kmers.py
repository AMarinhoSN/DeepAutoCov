import csv
import os

def calculate_kmers(sequences, k):
    # This function is designed to calculate k-mers from a list of sequences.
    # A k-mer is a substring of length 'k' derived from a longer sequence.
    # The function takes two parameters:
    # sequences: A list of sequences (strings) from which to derive the k-mers.
    # k: The length of each k-mer.

    # Initialize an empty list to store all k-mers.
    kmers = []

    # Iterate over each sequence in the provided list.
    for sequence in sequences:
        # Loop through the sequence to extract all possible k-mers.
        # The range is set up to stop at a point where a full-length k-mer can be obtained.
        for i in range(len(sequence) - k + 1):
            # Extract the k-mer starting at the current position 'i' and spanning 'k' characters.
            kmer = sequence[i:i+k]
            # Append the extracted k-mer to the list of kmers.
            kmers.append(kmer)

    # Return the list of all k-mers extracted from the input sequences.
    return kmers

def format_csv(seq, identifier, kmers_tot, k, week, l, path):
    # This function is designed to format data into a CSV file.
    # It takes the following parameters:
    # seq: The sequence from which k-mers are generated.
    # identifier: A unique identifier for the sequence.
    # kmers_tot: A list of all possible k-mers.
    # k: The length of each k-mer.
    # week: The week number, used in naming the output file.
    # l: A parameter not used in the function (possibly intended for future use).
    # path: The path to the directory where the CSV file will be saved.

    # Initialize lists for k-mers of the sequence and a binary representation.
    kmers = []
    binary = [identifier]  # Start the binary list with the sequence identifier.

    kmers.append(None)  # Append a placeholder at the start of the k-mers list.

    # Generate k-mers from the input sequence.
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        kmers.append(kmer)

    # Create a binary representation where 1 indicates the presence of a k-mer from kmers_tot in kmers.
    for km in kmers_tot:
        if km in kmers:
            binary.append(1)
        else:
            binary.append(0)

    # Add a placeholder at the beginning of kmers_tot.
    kmers_tot = [None] + kmers_tot

    # Write the data to a CSV file in the specified path.
    with open(str(path) + '/' + str(week) + '/' + str(identifier) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(kmers_tot)  # Write the total k-mers row.
        writer.writerow(binary)     # Write the binary representation row.

    return 'Done'  # Indicate that the function has completed its task.










