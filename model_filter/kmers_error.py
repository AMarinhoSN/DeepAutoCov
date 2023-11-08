import numpy as np
import csv

def kmers_importance(prediction, true_sequence, kmers):
    differences = []
    for i in range(len(prediction)):
        seq_pred = prediction[i]
        seq_real = true_sequence[i]
        differences.append(seq_pred - seq_real)
    sorted_indices = sorted(range(len(differences)), key=lambda k: differences[k], reverse=True)
    top_6_indices = sorted_indices[:6]
    kmers_selected = [kmers[i] for i in top_6_indices]

    return kmers_selected

def selection_kmers(AE_prediction, True_sequences, kmers, AE_classes, identifier, output_filename="summary_KMERS.csv"):
    ## INPUT
    # AE_PREDICTION: sequences predictions provided by the model [list];
    # True_sequence: real sequences encoded in the k-mers space [list];
    # kmers: kmers filtered (kmers that are different from zero by at least 0.01 %) defined in the script "Main_prediction_AE" as "features_no_zero" [list];
    # AE_classes: class defined by the model (1(normal),-1(anomaly)) [nparray];
    # identifier: ID_Sequence [list];
    # output_filename: path/name of file csv ["string"].

    ## OUTPUT
    # file CSV :
    #   1) The first column contain the id_sequence;
    #   2) The other columns contain the k-mers;

    anomaly_indices = np.where(AE_classes == -1)[0]
    AE_prediction_anomalies = [AE_prediction[i] for i in anomaly_indices]
    True_sequences_anomalies = [True_sequences[i] for i in anomaly_indices]
    identificativo_anomalies = [identifier[i] for i in anomaly_indices]
    summary = []

    for i in range(len(AE_prediction_anomalies)):
        prediction = AE_prediction_anomalies[i]
        real = True_sequences_anomalies[i]
        kmers_selected = kmers_importance(prediction, real, kmers)
        summary.append([identificativo_anomalies[i]] + kmers_selected)

    # Writing the summary to a CSV file
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing the header
        header = ['Identificativo'] + ['kmer_' + str(i+1) for i in range(len(summary[0])-1)]
        csvwriter.writerow(header)
        # Writing the rows
        csvwriter.writerows(summary)


# # Mock Data
# AE_prediction = [
#     [0.1, 0.2, 0.3, 0.4, 0.5],
#     [0.5, 0.4, 0.3, 0.2, 0.1],
#     [0.2, 0.3, 0.4, 0.5, 0.6]
# ]
# True_sequences = [
#     [0.1, 0.3, 0.3, 0.4, 0.5],
#     [0.4, 0.4, 0.3, 0.2, 0.1],
#     [0.2, 0.3, 0.5, 0.5, 0.7]
# ]
# kmers = ["A", "T", "C", "G", "N"]
# AE_classes =np.array([1, -1, -1])
# identificativo = ["ID1", "ID2", "ID3"]
# week = "Week1"
#
# # Call the function
# selection_kmers(AE_prediction, True_sequences, kmers, AE_classes, identificativo)
