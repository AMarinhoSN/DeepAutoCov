from keras.models import load_model
import pandas as pd
import numpy as np
import csv

def read_list_from_file(fpath):
    try:
        with open(fpath, 'r') as file:
            # read every row of the file and create a list with rows as elementes
            row_list = [line.strip() for line in file]
            return row_list
    except FileNotFoundError:
        print(f"file '{fpath}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def read_fasta_seq_ID(file_path):
    """
    Reads a FASTA file and returns two lists:
    1) list with sequence IDs
    2) list with the sequences.

    :param file_path: Path to the FASTA file to read.
    :return: A tuple of two lists, (list_of_sequence_ids, list_of_sequences).
    """
    sequence_ids = []
    sequences = []
    with open(file_path, 'r') as f:
        current_sequence = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):  # Line is a sequence identifier
                if current_sequence:  # If there's a current sequence, save it
                    sequences.append(current_sequence)
                    current_sequence = ''
                sequence_ids.append(line[1:])  # Save the ID without the '>'
            else:
                current_sequence += line
        if current_sequence:  # Don't forget to save the last sequence
            sequences.append(current_sequence)
    return sequence_ids, sequences

def calculate_kmers(sequences, k):
    kmers = []
    for sequence in sequences:
        for i in range(len(sequence)-k+1):
            kmer = sequence[i:i+k]
            kmers.append(kmer)
    return kmers

def format_csv(seq,identificativo,kmers_tot,k,week,l,path):

    kmers=[]
    binary=[]
    binary.append(identificativo)
    kmers.append(None)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        kmers.append(kmer)
    for i,km in enumerate(kmers_tot):
        if kmers_tot[i] in kmers:
            binary.append(1)
        else:
            binary.append(0)
    kmers_tot=[None]+kmers_tot
    with open(str(path)+'/'+str(week)+'/'+str(identificativo)+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(kmers_tot)
        writer.writerow(binary)
    return 'Done'


def kmer_presence(sequence_kmers, total_kmers):
    """
    Given a vector of k-mers of a sequence and a vector of total k-mers,
    returns a sequence of 0s and 1s indicating the presence or absence of each k-mer
    of the vector total_kmers in the sequence_kmers set.

    :param sequence_kmers: List of k-mers in the sequence.
    :param total_kmers: List of total k-mers.
    :return: List of 0's and 1's representing the presence/absence of k-mers.
    """
    # initialize a list of 0 for each kmer
    presence_sequence = [0] * len(total_kmers)

    # Imposta 1 se il k-mer è presente in sequence_kmers
    for i, kmer in enumerate(total_kmers):
        if kmer in sequence_kmers:
            presence_sequence[i] = 1

    return presence_sequence

def predict(data, threshold,model):
    input = data
    Autoencoders = load_model(str(model))
    # prediction
    output = Autoencoders.predict(input)
    # compute mse
    mse = np.mean(np.power(input - output, 2), axis=1)
    mse_list = list(mse)
    error_df = pd.DataFrame({'Reconstruction_error': mse})
    y_test_i_predict = [-1 if e >= threshold else 1 for e in error_df.Reconstruction_error.values]
    prediction = list(y_test_i_predict)

    return mse_list, prediction, output

def remove_asterisks(sequence):
    return sequence.rstrip("*")


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

def selection_kmers(AE_prediction, real_sequences, kmers, AE_classes, identifier):
    """

    :param AE_prediction:  sequences predictions provided by the model [list];
    :param real_sequences: real sequences encoded in the k-mers space [list];
    :param kmers: kmers filtered (kmers that are different from zero by at least 0.01 %) defined in the script "Main_prediction_AE" as "features_no_zero" [list];
    :param AE_classes: class defined by the model (1(normal),-1(anomaly)) [nparray];
    :param identifier: ID_Sequence [list];
    :return: out_dict: dict. keys: id_sequences, value:list of kmers
    """

    anomaly_indices = np.where(np.array(AE_classes) == -1)[0]
    AE_prediction_anomalies = [AE_prediction[i] for i in anomaly_indices]
    real_sequences_anomalies = [real_sequences[i] for i in anomaly_indices]
    identificativo_anomalies = [identifier[i] for i in anomaly_indices]
    out_dict = {}

    for i in range(len(AE_prediction_anomalies)):
        prediction = AE_prediction_anomalies[i]
        real = real_sequences_anomalies[i]
        kmers_selected = kmers_importance(prediction, real, kmers)
        out_dict[identificativo_anomalies[i]] = kmers_selected
        # summary.append([identificativo_anomalies[i]] + kmers_selected)

    return out_dict
    # Writing the summary to a CSV file
    """
        with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing the header
        header = ['Seq ID'] + ['k-mer_' + str(i+1) for i in range(len(summary[0])-1)]
        csvwriter.writerow(header)
        # Writing the rows
        csvwriter.writerows(summary)
        
    """
