from optparse import OptionParser
import sys
from utils import *
import json

def main(options):
    output_path = str(options.path_save)

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
    misrep_kmers = selection_kmers(outputs, dataset_tot, total_kmers , prediction, sequence_ids)

    out_dict = {}
    for i in range(0,len(mse_list)):
        # summary.append([sequence_ids[i],mse_list[i],prediction[i]])
        info_dict = {}
        if sequence_ids[i] in misrep_kmers.keys():
            info_dict["misrepresented_kmers"] = misrep_kmers[sequence_ids[i]]
        info_dict["is_anomaly"] = prediction[i]
        info_dict["anomaly_score"] = mse_list[i]
        out_dict[sequence_ids[i]] = info_dict

    # create json
    with open(output_path, 'w') as file:
        json.dump(out_dict, file)
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
    parser.add_option("-o", "--pathsave ", dest="path_save",
                      help="path of output json file",
                      default='')
    (options, args) = parser.parse_args()
    main(options)
