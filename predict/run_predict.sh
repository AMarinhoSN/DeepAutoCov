#!bin/bash

mydir="/home/bms-bioinfo/simone"
cd $mydir
fasta_file="/home/bms-bioinfo/simone/Spikes_prova.fasta"
model="/home/bms-bioinfo/simone/Autoencoder_models.h5"
feat="/home/bms-bioinfo/simone/cleaned_list.txt"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate deepautocov_env

python predict_new_samples.py -p $fasta_file -k 3 -s $mydir -f $feat -m $model