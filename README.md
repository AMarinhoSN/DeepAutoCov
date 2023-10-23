# VarAnomaly
In the folder "model" are contained the script of Deep Learning model and the simulation. 
In the folder "CreationDataset" are contained the script to create the dataset and feature representations

# Creation dataset
The main file to create the dataset is "Data_filtration_kmers.py". This code wants in input : 
1. -f path where the file fasta is stored (In the drive "data_github" the file is Spikes_prova.fasta); 
2. -c path where the file csv is stored (In the drive "data_github" the file is pseudodataset.csv);
3. -n nation if not made explicit takes all the data in the csv file;
4. -m length minimum; 
5. -l range; 
6. -p path where it’s possible save the file.
like outputs the code create a drive (call as you have defined example "dataset_interest_2023") where the sequences are stored and the meetadata filtered (In the file csv_dataset.py it's possible decide the name of file filtered) 

To run the code:
python Data_Filtration_kmers.py -f Spikes_prova.fasta -c pseudodataset.csv -m 1000 -l 30 -p /path/to/save/dataset_interest_2023

# Model
The main file to create the dataset is "Data_filtration_kmers.py". This code wants in input : 





