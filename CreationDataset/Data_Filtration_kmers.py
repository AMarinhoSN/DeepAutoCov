# Import the function
from optparse import OptionParser
from read_fasta import *
from read_csv import *
from save_sequence import *
from elimina_asterisco import *
from Filtra_lunghezze import *
import statistics as st
from Kmers import *
from data_time import *
from sort_metadata import *
import os
from csv_dataset import *

def main(options):
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


        print('\033[1m filter the sequences by the length included in the median \033[0m')
        extreme_inf = st.median(Length_filtering_min_1000_valid) - options.median_limit
        extreme_sup = st.median(Length_filtering_min_1000_valid) + options.median_limit
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


        write_csv_dataset(metadata, l)


        print('\033[1m Calculation of k-mers\033[0m')
        k = 3
        kmers = calculate_kmers(valid_sequence, k)
        kmers_unici = list(set(kmers))

        for i in range(0, len(indices_by_week)):
            indices = indices_by_week[i]
            sequences_for_week = []
            identificativo_for_week = []
            week = i + 1
            os.makedirs(str(options.save_path) + l + '/' + str(week))
            for m, index in enumerate(indices):
                sequences_for_week.append(sequences[index])
                identificativo_for_week.append(metadata[index, 4])
            for h, seq in enumerate(sequences_for_week):
                format_csv(seq, identificativo_for_week[h], kmers_unici, k, week, l)

        # Creating the dataset
        import os
        import csv

        csv_directory = str(options.save_path) + l

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
    parser = OptionParser()

    parser.add_option("-f", "--fasta", dest="fasta_path",

                      help="path to FASTA file", default="spikes.fasta")
    parser.add_option("-c", "--csv", dest="csv_path",

                      help="path to CSV file", default="metadata.csv")
    parser.add_option("-n","--continent",dest="continent_list",
                      help="list of continents of interest",default=['/'])

    parser.add_option("-m", "--minlen ", dest="min_length",
                      help="minimum length of sequence", default=1000)

    parser.add_option("-l", "--median_limit ", dest="med_limit",
                      help="median range", default=30)

    parser.add_option("-p", "--path_salvataggio_file ", dest="save_path",
                      help="path where saving the file", default='/blue/salemi/share/varcovid/dataset_febb_2023_')


    (options, args) = parser.parse_args()
    main(options)

# python your_script_name.py -f /path/to/spikes.fasta -c /path/to/metadata.csv