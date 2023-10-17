import csv
import os

def calculate_kmers(sequences, k):
    kmers = []
    for sequence in sequences:
        for i in range(len(sequence)-k+1):
            kmer = sequence[i:i+k]
            kmers.append(kmer)
    return kmers

def format_csv(seq,identificativo,kmers_tot,k,week,l):

    kmers=[]
    binary=[]
    binary.append(identificativo)
    kmers.append(None)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        kmers.append(kmer)
    for i,km in enumerate(kmers_tot):
        if kmers_tot[i] in kmers:
            binary.append(1)
        else:
            binary.append(0)
    kmers_tot=[None]+kmers_tot
    #os.makedirs('/Users/utente/Desktop/Varcovid/Nuovi_dati/'+str(week))
    with open('/blue/salemi/share/varcovid/dataset_febb_2023_'+l+'/'+str(week)+'/'+str(identificativo)+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(kmers_tot)
        writer.writerow(binary)
    return 'fatto'


# # example of use:
# sequenze_valide = ["MKTITLEVEDEPGSLYEEDKVLLSVAPQDSGPAVGRQLGVRISGKVFKDVNRLVRVVDGKT", "MKLIPTFTVGGPGMGLLSAFAPTSQAKLATDKYHNLFTYTRVLPIGMEYLPPEHVWQTFT", "MKVAHLTPATLPPLPSQTNRVIQYNNYQSAGGPYTLTMFLLSESIYTENGQWQVSDMNPL"]
# k = 3
# kmers = calculate_kmers(sequenze_valide, k) #Clcolo tutti i kmer contenuti nel datbase
# kmers_unici = list(set(kmers)) # ho la lista dei kmer unici
# seq="MKTITLEVEDEPGSLYEEDKVLLSVAPQDSGPAVGRQLGVRISGKVFKDVNRLVRVVDGKT"
# bin,kmer=format_csv(seq,'A23456',kmers_unici,k,34)







