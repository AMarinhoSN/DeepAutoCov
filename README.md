# DeepAutoCov
Code and data to reproduce the simulation in the following publication:

_Forecasting dominance of SARS-CoV-2 lineages by anomaly detection using deep AutoEncoders
Simone Rancati, Giovanna Nicora, Mattia Prosperi, Riccardo Bellazzi, Simone Marini, Marco Salemi
bioRxiv 2023.10.24.563721; doi: https://doi.org/10.1101/2023.10.24.563721_

Scripts to predict anomalies, i.e., Future Dominant Lineages (**FDLs**) with the Deep Learning AutoEncoder and to perform the simulation are located in the <code>model</code> folder.
Scripts to generate the dataset and the feature representations are within the <code>Feature_Extraction</code> folder.

## Feature Extraction
The file to create the dataset is <code>Data_filtration_kmers.py</code>. Example:
<code>python Data_Filtration_kmers.py -f Spikes_prova.fasta -c pseudodataset.csv -m 1000 -l 30 -p /path/to/save/dataset_interest_2023 </code>

Mandatory
-f: path where the input fasta file is stored (Example file: <code>data_github/Spikes_prova.fasta</code>).
-c: path where the input metadata (csv) is stored (Example file: <code>data_github/pseudodataset.csv</code>). Sequences and metadata should be in the same order. All columns are necessary and must be in the same order as in the example file, i.e.: <code> Virus name, Last vaccinated, Passage details/history, Type, Accession ID, Collection date, Location, Additional location information, Sequence length, Host, Patient age, Gender, Clade, Pango lineage, Pango version, Variant, AA Substitutions, Submission date, Is reference?, Is complete?, Is high coverage?, Is low coverage?, N-Content, GC-Content</code>

Optional
-n: nation (e.g., "France") (if not specified, all sequences are used) (<code>default: ['/']</code>);
-m: Filter: minimum lenght of the spike sequences (<code>default value: 1000</code>); 
-l: Filter: accepted amino acid distant from lineage median (<code>default value: 30</code>); as in: for each lineage, how the protein length can vary to be accepted?
-p: path where it’s possible save the file.
The output is a folder (for example "dataset_interest_2023") where the sequences are stored (in the csv and text format) and the metadata of the filtered sequences (In the file csv_dataset.py it's possible decide the name of file filtered) 


-Output:
1) CSV File: Contains the information of the filtered sequences;
2) Dataset: It creates a folder that in turn contains subfolders (numbered by weeks) that contain:
  a) file csv for each sequence, in the first raw contains the kmers,in the second contains a sequence of 0/1 that indicates the presence or absence of kmers
  b) file.txt that contains the identificators and the sequence of 0/1

## Model prediction
To predict anomalies, you can use the script <code>Main_prediction_AE.py</code>. The script takes as input : 
1. -p path where the sequences are located (/path/to/save/dataset_interest_2023/);
2. -c path where the file filtered csv is storerd (the metadata file filtered created in the "Data_filtration_kmers.py");
3. -k path where the possible kmers are stored (example is the first line of csv file created in dataset_interest_2023);
4. -s path to saving the file;
5. -m fraction of kmers to mantain (<code>default value: 0.05</code>);
6. -e number of epochs (<code>default value: 300</code>);
7. -b batch size for the first week (<code>default value: 256</code>);
8. -d Sets the encoding dimension (<code>default value: 1024</code>);
9. -r learning rate (<code>default value: 1e-7</code>).

To run the code:
<code>python Main_prediction_AE.py -p /path/to/drive -c /path/to/metadata.csv -k /path/to/kmers_file.csv -s /path/where/to/save/output -m 0.1 -e 300 -b 256 -d 1024 -r 1e-7 </code>

-Output:
1) Precision-graph of the top 100 sequences considereted like anomalies by DeepAutoCov model (<code>Fraction_general100</code>);
2) file.log containing for each week of simulation how many sequences the model identified like anomalies for each Future Dominant Lineage or FDL (<code>Autoencode_performance.log</code>);
3) Graph of the precision considering all the sequences considerated anomalies by DeepAutoCov model;
4) Graph F1,Precision,Recall ( these graphs are as tests to see how the model was doing not considering the fact that the "Anomaly" class varies each time ); 
5) File.h5 which contains the information (weights) of the trained autoencoder;
6) Graph of the trend of the number of features over time;
7) Graph of how many weeks in advance ndividing the FDLs with respect to the threshold;
8) file CSV that contains for each sequence analysed the k-mers not reproduced correctly (the first column contain the id_sequence and other columns contain the k-mers)






