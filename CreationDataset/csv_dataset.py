import numpy as np
import csv


def write_csv_dataset(array,l):
    # Definizione dei nomi delle colonne come lista di stringhe
    nomi_colonne = ['Virus.name','Not.Impo','format','Type','Accession.ID','Collection.date','Location','Additional.location.information','Sequence.length','Host','Patient.age','Gender','Clade','Pango.lineage','Pangolin.type','Variant','AA.Substitutions','Submission.date','Is.reference.','Is.complete.','Is.high.coverage.','Is.low.coverage.','N.Content']
    # Apertura del file CSV in modalità scrittura e definizione del writer
    with open('/blue/salemi/share/varcovid/filtered_metadatataset_010223_edit_020523_'+l+'.csv', "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        # Scrittura della riga d'intestazione con i nomi delle colonne
        writer.writerow(nomi_colonne)

        # Scrittura delle righe dei dati
        for riga in array:
            writer.writerow(riga)