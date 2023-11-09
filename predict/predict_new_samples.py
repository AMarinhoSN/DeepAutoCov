from optparse import OptionParser
import sys
from utils import *

def main(options):
    path_salvataggio_file = str(options.path_save)

    print('Read the FASTA file of sequence')
    sequence_ids, sequences = read_fasta_seq_ID(str(options.fasta_path))
    sequences = [remove_asterisks(s) for s in sequences]

    if not sequences:
        print('The FASTA file is empty. Please upload an other FASTA file')
        sys.exit()

    print('Compute the k-mers')
    k = int(options.kmers) # set the length of k-mers = 3
    kmers_sequences_total = []
    for i in range(len(sequences)):
        seq = [sequences[i]]
        kmers_sequence = calculate_kmers(seq,k)
        kmers_sequences_total.append(kmers_sequence)

    # total k-mers is a database and model specific
    total_kmers = read_list_from_file(str(options.features_path))
    total_kmers = list(total_kmers)

    dataset_tot = []
    for i in range(len(kmers_sequences_total)):
        data_set = kmer_presence(kmers_sequences_total[i],total_kmers)
        dataset_tot.append(data_set)

    threshold = options.thr

    print('Prediction')
    mse_list, prediction, outputs = predict(dataset_tot, threshold,str(options.model))
    selection_kmers(outputs, dataset_tot, total_kmers , prediction, sequence_ids, output_filename=path_salvataggio_file+"/summary_KMERS.csv")

    summary = []
    for i in range(0,len(mse_list)):
        summary.append([sequence_ids[i],mse_list[i],prediction[i]])

    # create txt
    with open(path_salvataggio_file + '/prediction_seq.txt', 'w') as file:
        # Scrivi ogni elemento della lista in una nuova riga nel file
        file.write('#Legend: -1:Anomaly, 1:Not Anomaly' + '\n')
        file.write('Seq_ID, Anomaly_Score, Anomaly' + '\n')
        for elemento in summary:
            file.write(str(elemento[0]) + ','+str(round(elemento[1],3))+','+str(elemento[2]) + '\n')

    print('DONE!!')
if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-p", "--pathdrive", dest="fasta_path",

                          help="path to fasta file: path/fasta",
                          default="")  # default

    parser.add_option("-f", "--pathfeatures", dest="features_path",
                      help="path to txt features",
                      default="")  # default

    parser.add_option("-k", "--kmers", dest="kmers",
                          help='kmers',
                          default= 3)

    parser.add_option("-m", "--model ", dest="model",
                          help="path model",
                          default='')

    parser.add_option("-t", "--threshold ", dest="thr",
                      help="threshold",
                      default=0.03)
    parser.add_option("-s", "--pathsave ", dest="path_save",
                      help="path where we can save the file",
                      default='')
    (options, args) = parser.parse_args()
    main(options)
