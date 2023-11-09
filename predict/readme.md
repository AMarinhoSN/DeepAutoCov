# Readme

This folder contains script and sample data to predict anomalies in a fasta file.

## Files description
- [samples_spike.fasta](samples_spike.fasta): example of a fasta file (input for the prediction)
- [features.txt](features.txt): txt file that contains the relevant kmers for the trained model
- [predict_new_samples.py](predict_new_samples.py): python script to predict the sequences in .fasta file
- [utils.py](utils.py): relevant functions for prediction
- [run_predict.sh](run_predict.sh): example of a bash script to run the prediction (see also commands below)

## Requirements and installation
You can create a conda env with all the required packages thanks to the [deepautocov_env.yml](predict/deepautocov_env.yml) file

<code>conda env create -f deepautocov_env.yml</code>

To activate the conda env:

<code>conda activate deepautocov_env</code>

## Usage
To predict the anomalies in the fasta file, run the prediction script as follows:

<code>python predict_new_samples.py -p samples_spike.fasta -o /path/to/output/json -f features.txt -m Autoencoder_models.h5 </code>

### Arguments
 <code>-p</code>: input fasta file
 
 <code>-k</code>: kmer length
 
 <code>-f</code>: features list in a txt file (see features.txt file). This file is generated together with the h5 file during training
 
 <code>-m</code>: h5 file containing the autoencoder trained model
 
 <code>-o</code>: output json file path where predictions will be written
 
 
 ## Output
 JSON file containing for each sequence id the following information:
 - whether the sequence is predicted as anomaly (<code>is_anomaly</code>).
  If the value is -1, than the sequence is an anomaly
  - the <code>anomaly_score</code>
  - <code>misrepresented_kmers</code>: if the sequence is predicted as anomaly,
   this list contains the misrepresented kmers 
  ```

  {
   "EPI_ID1": 
     {"is_anomaly": 1, 
      "anomaly_score": 0.021185704234078836}, 
   "EPI_ID2": 
     {"is_anomaly": 1, 
      "anomaly_score": 0.023654659820759667}, 
   "EPI_ID2": 
    {"misrepresented_kmers": ["TVY", "NGI", "AQY"], 
     "is_anomaly": -1, 
     "anomaly_score": 0.36129919656940856}
}
  
  
  ```
 
 
