#!bin/bash

mydir="path/to/home"
cd $mydir
fasta_file="/path/to/Spikes_prova.fasta"
model="/path/to/Autoencoder_models.h5"
feat="/path/to/features.txt"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate deepautocov_env

python predict_new_samples.py -p $fasta_file -k 3 -s $mydir -f $feat -m $model
