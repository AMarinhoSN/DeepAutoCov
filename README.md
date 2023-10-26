# VarAnomaly
Within the <code>model</code> folder you can find the scripts to predict sequences with the Deep Learning model and to perform the simulation. 
Within the <code>Feature_Extraction</code> folder you can find the scripts to create the dataset and the feature representations

## Feature Extraction
The file to create the dataset is <code>Data_filtration_kmers.py</code>. These are the script's arguments: 
1. -f: path where the file fasta is stored (in the "data_github" Drive folder the file is <code>Spikes_prova.fasta</code>); 
2. -c: path where the file csv is stored (In the "data_github" Drive folder the file is <code>pseudodataset.csv</code>);
3. -n: nation (for instance "France") (if not made explicit takes all the data in the csv file) (<code>default: ['/']</code>);
4. -m: minimum lenght of the spike sequences (<code>default value: 1000</code>); 
5. -l: median length value that will be used to determine the minum and maximum lenght (<code>default value: 30</code>); 
6. -p: path where it’s possible save the file.
The output is a folder (for example "dataset_interest_2023") where the sequences are stored (in the csv and text format) and the metadata of the filtered sequences (In the file csv_dataset.py it's possible decide the name of file filtered) 

To run the code:
python Data_Filtration_kmers.py -f Spikes_prova.fasta -c pseudodataset.csv -m 1000 -l 30 -p /path/to/save/dataset_interest_2023

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
python Main_prediction_AE.py -p /path/to/drive -c /path/to/metadata.csv -k /path/to/kmers_file.csv -s /path/where/to/save/output -m 0.1 -e 300 -b 256 -d 1024 -r 1e-7





