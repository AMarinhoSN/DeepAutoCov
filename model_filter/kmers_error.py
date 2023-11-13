import numpy as np
import csv


def kmers_importance(prediction, true_sequence, kmers):
    """
    This function identifies the most important k-mers based on the differences between predicted and true sequences.

    Parameters:
    - prediction: A list of predicted sequence values.
    - true_sequence: A list of actual sequence values.
    - kmers: A list of k-mers.

    Returns:
    - kmers_selected: A list of the top 6 k-mers based on their importance.
    """

    # Initialize a list to store the differences between predicted and true sequence values.
    differences = []

    # Iterate over the sequences and compute the difference between each predicted and true value.
    for i in range(len(prediction)):
        seq_pred = prediction[i]
        seq_real = true_sequence[i]
        differences.append(seq_pred - seq_real)

    # Sort the indices based on the differences, in descending order (largest differences first).
    sorted_indices = sorted(range(len(differences)), key=lambda k: differences[k], reverse=True)

    # Select the indices corresponding to the top 6 differences.
    top_6_indices = sorted_indices[:6]

    # Retrieve the k-mers corresponding to these top 6 indices.
    kmers_selected = [kmers[i] for i in top_6_indices]

    # Return the selected k-mers.
    return kmers_selected


def selection_kmers(AE_prediction, True_sequences, kmers, AE_classes, identifier, output_filename="summary_KMERS.csv"):
    """
    This function selects k-mers based on anomalies detected by an autoencoder model and outputs the results to a CSV file.

    Inputs:
    - AE_prediction: List of sequences predicted by the model.
    - True_sequences: List of real sequences encoded in the k-mers space.
    - kmers: List of filtered k-mers that differ from zero by at least 0.01%.
    - AE_classes: Numpy array of classes defined by the model (1 for normal, -1 for anomaly).
    - identifier: List of sequence IDs.
    - output_filename: String specifying the path and name of the output CSV file.

    Output:
    - A CSV file with the following columns:
        1) The first column contains the sequence ID.
        2) The other columns contain the selected k-mers.
    """

    # Identify indices where the model has classified the sequences as anomalies.
    anomaly_indices = np.where(AE_classes == -1)[0]

    # Extract predictions, true sequences, and identifiers for the anomalies.
    AE_prediction_anomalies = [AE_prediction[i] for i in anomaly_indices]
    True_sequences_anomalies = [True_sequences[i] for i in anomaly_indices]
    identificativo_anomalies = [identifier[i] for i in anomaly_indices]

    # Initialize a list to store the summary data.
    summary = []

    # Iterate over each anomaly to select k-mers and compile the summary information.
    for i in range(len(AE_prediction_anomalies)):
        prediction = AE_prediction_anomalies[i]
        real = True_sequences_anomalies[i]

        # Determine the importance of k-mers based on the prediction and real sequence.
        kmers_selected = kmers_importance(prediction, real, kmers)

        # Append the sequence identifier and the selected k-mers to the summary.
        summary.append([identificativo_anomalies[i]] + kmers_selected)

    # Write the summary to a CSV file.
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the header of the CSV file.
        header = ['Identificativo'] + ['kmer_' + str(i + 1) for i in range(len(summary[0]) - 1)]
        csvwriter.writerow(header)

        # Write each row of the summary to the CSV file.
        csvwriter.writerows(summary)



