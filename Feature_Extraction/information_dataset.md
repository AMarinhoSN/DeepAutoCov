In this section i report the script to craeate  a dataset for our model.
The main script is <code>Data_Filtration_kmers.py</code> where the data is filtered and the dataset is created. In main script several functions are called, in particular:
1) <code>read_fasta.py</code>: to read a fasta file;
2) <code>read_csv.py</code>: to read a csv file;
3) <code>save_sequence.py</code>: to filter the correct sequences;
4) <code>elimina_asterisco.py</code>: to eliminate the asterisk after the sequence;
5) <code>Filtra_lunghezze.py</code>: to filter the length of sequences;
6) <code>Kmers.py</code>: to calculate the kmers of sequences;
7) <code>sort_metadata.py</code>: to reorder the sequences;
8) <code>csv_dataset.py</code>: to transform dataset in csv.
