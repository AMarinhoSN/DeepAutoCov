import numpy as np
import csv

def write_csv_dataset(array, l, path_to_save):
    # This function writes a dataset to a CSV file.
    # Parameters:
    # array: The dataset to be written, assumed to be a collection of rows (like a list of lists or a 2D array).
    # l: A label or identifier used in the naming of the output CSV file.
    # path_to_save: The file path where the CSV file will be saved.

    # Define the column names for the CSV file as a list of strings.
    name_columns = ['Virus.name', 'Not.Impo', 'format', 'Type', 'Accession.ID',
                    'Collection.date', 'Location', 'Additional.location.information',
                    'Sequence.length', 'Host', 'Patient.age', 'Gender', 'Clade',
                    'Pango.lineage', 'Pangolin.type', 'Variant', 'AA.Substitutions',
                    'Submission.date', 'Is.reference.', 'Is.complete.', 'Is.high.coverage.',
                    'Is.low.coverage.', 'N.Content']

    # Open a new CSV file in write mode at the specified path.
    with open(path_to_save + '/filtered_metadatataset_' + l + '.csv', "w", newline="") as csvfile:
        # Create a CSV writer object with comma as the delimiter.
        writer = csv.writer(csvfile, delimiter=",")

        # Write the first row of the CSV file using the column names.
        writer.writerow(name_columns)

        # Iterate through each row in the input array and write it to the CSV file.
        for row in array:
            writer.writerow(row)