# Import builtin python function
import argparse
import os
import csv
from datetime import datetime
import statistics as st
import random

# import external libraries
import pandas as pd
import numpy as np


# --- FUNCTIONS --- #
def read_fasta(file):
    sequences = []
    with open(file, 'r') as f:
        current_sequence = ''
        started = False
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if started:
                    sequences.append(current_sequence)
                started = True
                current_sequence = ''
            else:
                current_sequence += line
        if current_sequence:
            sequences.append(current_sequence)
    return sequences

def read_csv(file):
    return pd.read_csv(file).values

def read_tsv(file):
    return pd.read_csv(file,sep='\t').values

# filtra_sequenze: function to filter the length of sequences 
# INPUT:
#    1) sequences: list
#    2) length_minimum : minimum length acceptable 
#    3) length_maximum : maximum length acceptable 
def filtra_sequenze(sequences, length_minimum, length_maximum): # We ecide the maximum and minimum length i.e. I put the general median - 20 amino acids and + 20 amino acids 
    index = [i for i, seq in enumerate(sequences) if length_minimum <= len(seq) <= length_maximum]
    sequences_valid = [seq for i, seq in enumerate(sequences) if length_minimum <= len(seq) <= length_maximum]
    return index, sequences_valid


def validate_sequences(sequences):
    """
    validate_sequences: function to find the correct sequences
    """
    
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

def remove_asterisks(sequence):
    return sequence.rstrip("*")

# --- KMERS ---
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
    return 'fatto'



# sequenza=['ASTREFGIHILMONOPRST','ASTREFGIHILMONOPRST','A','BVG','ASTREFGIHILMONOPRST']
# indici,sequenze_valide=filtra_sequenze(sequenza, 4, 22)
# print(indici)
# print(sequenze_valide)

# --- DATE TIME ---
def split_weeks(dates):
    date_objs = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    date_objs.sort()
    start_date = date_objs[0]
    end_date = date_objs[-1]

    num_weeks = ((end_date - start_date).days // 7) + 1
    indices_by_week = [[] for _ in range(num_weeks)]

    for i, date_obj in enumerate(date_objs):
        days_diff = (date_obj - start_date).days
        week_num = days_diff // 7
        indices_by_week[week_num].append(i)

    return indices_by_week

def trimestral_indices(dates_list,m):
    # Converti le date in oggetti datetime
    dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates_list]

    # Crea un dizionario che associa a ogni trimestre (anno, trimestre) la lista degli indici delle date in quel trimestre
    trimestral_indices = {}
    for i, date in enumerate(dates):
        year = date.year
        trimester = (date.month - 1) // m + 1
        key = (year, trimester)
        if key not in trimestral_indices:
            trimestral_indices[key] = []
        trimestral_indices[key].append(i)

    # Restituisci la lista di liste degli indici dei trimestri, ordinati per anno e trimestre
    sorted_keys = sorted(trimestral_indices.keys())
    return [trimestral_indices[key] for key in sorted_keys]

# --- SORT METADATA ---
def filter_row_by_column_length(ndarray, string_list, col_index, target_len):
    mask = np.array([len(s) == target_len for s in ndarray[:,col_index]], dtype=bool)
    return ndarray[mask], [s for s, m in zip(string_list, mask) if m], ndarray[np.logical_not(mask)], [s for s, m in zip(string_list, mask) if not m]

# filter_row_by_column_length_sostitution : find the sequences correctly with corrects date 
def filter_row_by_column_length_sostitution(ndarray, string_list, col_index, target_len):

    def check_and_add_suffix(s):
        suffixes = ["-01", "-07", "-10", "-14", "-20", "-25", "-27"]
        if len(s) == 7:
            extended_s = s + random.choice(suffixes)
            if len(extended_s) == target_len:
                return extended_s
        return s

    extended_strings = np.array([check_and_add_suffix(s) for s in ndarray[:, col_index]])
    mask = np.array([len(s) == target_len for s in extended_strings], dtype=bool)
    ndarray[:, col_index] = extended_strings

    return ndarray[mask], [s for s, m in zip(string_list, mask) if m], ndarray[np.logical_not(mask)], [s for s, m in zip(string_list, mask) if not m]

def insert_sequence_as_column(data, dates, sequence):
    """
    It inserts the amino acid sequence as a column of an ndarray that contains the metadata,
    then sorting the ndarray by dates (in ascending order).

    Args:
        data (ndarray): the ndarray containing the metadata
        dates (list): the list of dates corresponding to each row in the ndarray.
        sequence (list): the list of the amino acid sequence to be entered as a column.

    Returns:
        ndarray: the ndarray sorted by dates, with the amino acid sequence inserted as a column.
    """
    # Transform date in object datetime
    date_objs = np.array([datetime.strptime(date, '%Y-%m-%d') for date in dates])

    # Add the sequence column as the last column of the ndarray
    data_with_sequence = np.column_stack((data, sequence))

    # Add a column with datetime objects of the dates
    data_with_dates = np.column_stack((data_with_sequence, date_objs))

    # Sort the ndarray by dates
    sorted_indices = np.argsort(data_with_dates[:, -1])
    sorted_data = data_with_dates[sorted_indices]

    # Delete the date column
    sorted_data = np.delete(sorted_data, -1, axis=1)

    return sorted_data

def select_rows(ndarray,country):
    selected_rows = []
    for row in ndarray:
        if isinstance(row[6], str) and country in row[6]:
            selected_rows.append(row)
    return np.array(selected_rows)

# to create dataset
def select_rows_dataset(ndarray, country):
    selected_rows = []
    selected_indices = []

    for index, row in enumerate(ndarray):
        if isinstance(row[6], str) and country in row[6]:
            selected_rows.append(row)
            selected_indices.append(index)

    return np.array(selected_rows), selected_indices


# --- CSV DATASET ---

def write_csv_dataset(array,l):
    # Definition of column names as a list of strings.
    nomi_colonne = ['Virus.name','Not.Impo','format','Type','Accession.ID','Collection.date','Location','Additional.location.information','Sequence.length','Host','Patient.age','Gender','Clade','Pango.lineage','Pangolin.type','Variant','AA.Substitutions','Submission.date','Is.reference.','Is.complete.','Is.high.coverage.','Is.low.coverage.','N.Content']
    # Opening the CSV file in write mode and defining the writer.
    with open('filtered_metadatataset_'+l+'.csv', "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        # Writing header row with column names.
        writer.writerow(nomi_colonne)

        # Writing data rows
        for riga in array:
            writer.writerow(riga)

def feature_extractor_main(options):
    #Continent = ['Denmark', 'France', 'United Kingdom', 'USA', '/', 'Denmark']
    continent=options.continent_list
    for l in continent:
        # Read the file (metadata-> ndaaray, sequences-> list)
        print("\033[1m Read File Metadata and Fasta \033[0m") # METADATA IS A CVS of GISAID
        # sequences = read_fasta("/blue/salemi/share/varcovid/PAPAER_GROSSO/DATASET_NEI_PAESI/spikes.fasta")
        sequences = read_fasta(str(options.fasta_path))
        metadata = read_csv(str(options.csv_path))
        print("\033[1m Metadata file and Fasta have been uploaded \033[0m")

        # I eliminate the ending asterisk in some sequences. In the Fairy format, the asterisk is used to see where the sequence ends.        print("\033[1m Removal of final asterisk from Fasta format \033[0m")
        sequences = [remove_asterisks(s) for s in sequences]
        print(sequences)
        print("\033[1m Asterisks have been removed  \033[0m")


        print("\033[1m Filter lineages by country of provenance \033[0m")
        metadata_nation, index_nation = select_rows_dataset(metadata, l)
        sequences_nation = [sequences[i] for i in index_nation]
        print("\033[1m Lineages were filtered by country of provenance \033[0m")

        # Initial Dimension
        Dimension = len(sequences_nation)
        print('\033[1m The number of spike proteins : ' + str(Dimension) + '\033[0m')
        # Length delle sequenze
        Length = []
        for sequence in sequences_nation:
            Length.append(len(sequence))


        print('\033[1m Filter sequences with length less than ' + str(options.min_length) +'\033[0m')
        sequences_filtering_min_limit = [x for i, x in enumerate(sequences_nation) if Length[i] >= options.min_length]
        index = [i for i, x in enumerate(Length) if x >= options.min_length]
        print('\033[1m Update the Metadata file \033[0m')
        metadata_filter_min_limit = metadata_nation[index]
        Dimensione_fil_min_limit = len(sequences_filtering_min_limit)
        print('The number of spike proteins after deleting sequences with length less than '+ str(options.min_length)+' is: ' + str(
            Dimensione_fil_min_limit))

        # Calcolo la lunghezza nuova
        print('\033[1m Calculation of lengths filtered less than ' + str(options.min_length)+'\033[0m')
        Length_filtering_min_limit = []
        for sequence in sequences_filtering_min_limit:
            Length_filtering_min_limit.append(len(sequence))

        # Seleziono le Sequenze Valide
        print('\033[1m Evaluation of valid sequence contained in the database\033[0m')
        valid_sequences, invalid_sequences, valid_indices, invalid_indices = validate_sequences(
            sequences_filtering_min_limit)
        print('\033[1m Results : \033[0m')
        print("Ther'are " + str(len(valid_sequences)) + ' valid sequence in the database')
        print("Ther'are " + str(len(invalid_sequences)) + ' not valid sequence in the database')

        # Aggiorno il file metadata
        print('\033[1m Update the metadata  \033[0m')
        metadata_valid_indices = metadata_filter_min_limit[valid_indices]
        metadata_valid_invalid_indices = metadata_filter_min_limit[invalid_indices]

        Length_filtering_min_1000_valid = []
        for sequence in valid_sequences:
            Length_filtering_min_1000_valid.append(len(sequence))

        if not Length_filtering_min_1000_valid:
            print('There are not the valid sequences in the database. The analysis is stopped')
            break

        print('\033[1m filter the sequences by the length included in the median \033[0m')
        extreme_inf = st.median(Length_filtering_min_1000_valid) - int(options.med_limit)
        extreme_sup = st.median(Length_filtering_min_1000_valid) + int(options.med_limit)
        index, valid_sequence = filtra_sequenze(valid_sequences, extreme_inf, extreme_sup)
        metadata_valid_indices_length = metadata_valid_indices[index]
        print(str(len(valid_sequence)) +' sequences with length between ' + str(
            extreme_inf) + ' and ' + str(extreme_sup))

        print("\033[1m Filter sequences by dates \033[0m")
        metadata_off, sequences_off, metadata_not_off, sequences_not_off = filter_row_by_column_length_sostitution(
            metadata_valid_indices_length, valid_sequence, 5, 10)
        print("\033[1m The number of sequences filtered with dates is :\033[0m" + str(len(metadata_off)))

        print("\033[1m Reordering the metadata file \033[0m")
        metadata_tot = insert_sequence_as_column(metadata_off, metadata_off[:, 5], sequences_off)

        sequences = list(metadata_tot[:, 24])  # sequenze
        metadata = metadata_tot[:, 0:23]  # metadata


        print("\033[The number of simulation week : \033[0m")
        indices_by_week = split_weeks(metadata[:, 5])
        print(len(indices_by_week))
        seq_week = []
        for i in range(0, len(indices_by_week)):
            seq_week.append(len(indices_by_week[i]))
        print(seq_week)

        if l=='/':
            l='all'
        write_csv_dataset(metadata, l)


        print('\033[1m Calculation of k-mers\033[0m')
        k = 3
        kmers = calculate_kmers(valid_sequence, k)
        kmers_unici = list(set(kmers))

        import os
        for i in range(0, len(indices_by_week)):
            indices = indices_by_week[i]
            sequences_for_week = []
            identificativo_for_week = []
            week = i + 1
            os.makedirs(str(options.save_path) + '/' + str(week))
            for m, index in enumerate(indices):
                sequences_for_week.append(sequences[index])
                identificativo_for_week.append(metadata[index, 4])
            for h, seq in enumerate(sequences_for_week):
                format_csv(seq, identificativo_for_week[h], kmers_unici, k, week, l, str(options.save_path))

        # Creating the dataset

        csv_directory = str(options.save_path)

        # Loop attraverso tutte le sottodirectory e file nella cartella principale
        for root, directories, files in os.walk(csv_directory):
            for directory in directories:
                # Crea il file di testo e apri in modalità append
                txt_file = os.path.join(root, directory, "week_dataset.txt")
                with open(txt_file, "a") as output_file:
                    for filename in os.listdir(os.path.join(root, directory)):
                        if filename.endswith(".csv"):
                            csv_file = os.path.join(root, directory, filename)
                            # Apri il file CSV con la libreria csv e leggi ogni riga
                            with open(csv_file, "r") as input_file:
                                reader = csv.reader(input_file)
                                next(reader)  # salta la prima riga
                                for row in reader:
                                    # Scrivi ogni riga nel file di testo
                                    output_file.write(",".join(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature extraction script"
    )
    requiredNamed = parser.add_argument_group("required named arguments")
    requiredNamed.add_argument("-f", "--fasta", dest="fasta_path",
        help="path to a SPIKE FASTA file")

    requiredNamed.add_argument("-c", "--csv", dest="csv_path",
        help="path to GISAID metadata CSV file")

    parser.add_argument("-n","--continent",dest="continent_list",
                      help="list of continents of interest [DEFAULT=['/']]", default=['/'])

    parser.add_argument("-m", "--minlen ", dest="min_length",
                      help="minimum length of sequence", default=1000)

    parser.add_argument("-l", "--median_limit ", dest="med_limit",
                      help="median range", default=30)

    parser.add_argument("-o", "--output_dir", dest="save_path",
                      help="set ouput dir [DEFAULT=<current working directory>]", default=os.getcwd())

    args = parser.parse_args()
    feature_extractor_main(args)

# python3 Data_Filtration_kmers.py -f "Spikes_prova.fasta" -c "pseudodataset.csv" -p "/Users/utente/Desktop/PROVA_GITHUB" -l 30

# TODO: - fix arguments
# TODO: - add test data
# TODO: - 