import os
from keras.models import load_model
import pandas as pd
import numpy as np
import csv

def read_list_from_file(nome_file):
    try:
        with open(nome_file, 'r') as file:
            # Leggi ogni riga del file e crea una lista con le righe come elementi
            lista = [line.strip() for line in file]
            return lista
    except FileNotFoundError:
        print(f"file '{nome_file}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def read_fasta_seq_ID(file_path):
    """
    Reads a FASTA file and returns two lists: one with sequence IDs and the other with the sequences.

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
    #os.makedirs('/Users/utente/Desktop/Varcovid/Nuovi_dati/'+str(week))
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
    # Inizializza una lista di 0 per ogni k-mer in total_kmers
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
    output= Autoencoders.predict(input)
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

    anomaly_indices = np.where(np.array(AE_classes) == -1)[0]
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
        header = ['Seq ID'] + ['k-mer_' + str(i+1) for i in range(len(summary[0])-1)]
        csvwriter.writerow(header)
        # Writing the rows
        csvwriter.writerows(summary)