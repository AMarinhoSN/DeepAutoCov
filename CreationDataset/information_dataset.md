In this section i report the script to craeate  a dataset for our model.
The main script is Data_Filtration_kmers.py where the data is filtered and the dataset is created. In main script several functions are called, in particular:
1) read_fasta.py: to read a fasta file;
2) read_csv.py: to read a csv file;
3) save_sequence.py: to filter the correct sequences;
4) elimina_asterisco.py: to eliminate the asterisk after the sequence;
5) Filtra_lunghezze.py: to filter the length of sequences;
6) Kmers.py: to calculate the kmers of sequences;
7) sort_metadata.py: to reorder the sequences;
8) csv_dataset.py: to transform dataset in csv 
